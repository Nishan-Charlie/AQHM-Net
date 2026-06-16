"""
classical_backbone.py
---------------------
Classical feature-extraction backbone for AQHM-Net.

Architecture (Section 4 of METHODOLOGY.md):
  Stem -> UIB Stage1 + SE -> UIB Stage2 + CBAM -> UIB Stage3 + SE

Output shape: (B, 96, 7, 7) — one spatial cell per 4×4 patch of the
28×28 input, giving exactly 49 superpixel positions.

References:
  - MobileNetV4 UIB  : Qin et al. (2024)
  - SE attention     : Hu et al. (2018)
  - CBAM             : Woo et al. (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) Channel Attention
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention gate.

    Recalibrates channel-wise feature responses by squeezing spatial info
    into a channel descriptor and exciting (scaling) channels accordingly.

    Args:
        channels  : number of input/output channels.
        ratio     : bottleneck reduction ratio (default 0.25 -> C/4 hidden units).
                    Following Hu et al. (2018) recommendation.
    """

    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(1, int(channels * ratio))  # at least 1 hidden unit

        # Squeeze: global average pool collapses spatial dims -> (B, C)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Excitation: two FC layers with bottleneck
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            # Hardsigmoid is cheaper than sigmoid and used in MobileNetV3+
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, H, W)
        scale = self.gap(x)               # (B, C, 1, 1)
        scale = self.excite(scale)        # (B, C)
        # Broadcast multiply: rescale each channel map
        return x * scale.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# CBAM — Convolutional Block Attention Module
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """CBAM channel attention sub-module.

    Combines AvgPool- and MaxPool-based channel descriptors through a shared
    MLP, then gates the input feature map.  (Woo et al., 2018, Eq. 1-2)
    """

    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(1, int(channels * ratio))
        # Shared MLP — weight-tied for both pooling branches
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Average-pool and max-pool descriptors
        avg_desc = x.mean(dim=[2, 3])   # (B, C)
        max_desc = x.amax(dim=[2, 3])   # (B, C)

        # Pass each through the shared MLP then sum before sigmoid gate
        channel_att = torch.sigmoid(
            self.mlp(avg_desc) + self.mlp(max_desc)
        )  # (B, C)
        return x * channel_att.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial attention sub-module.

    Creates a 2-channel spatial descriptor (channel-wise avg + max), then
    produces a spatial gate via a 7×7 convolution.  (Woo et al., 2018, Eq. 3)

    The large 7×7 kernel is the recommended default in the original paper;
    it captures long-range spatial relationships.
    """

    def __init__(self) -> None:
        super().__init__()
        # Conv2d(in=2, out=1, kernel=7, padding=3) preserves spatial size
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise descriptors: (B, 1, H, W) each
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial_desc = torch.cat([avg_map, max_map], dim=1)  # (B, 2, H, W)
        spatial_att = torch.sigmoid(self.conv(spatial_desc))  # (B, 1, H, W)
        return x * spatial_att


class CBAMBlock(nn.Module):
    """Full CBAM: sequential channel then spatial attention.

    Placement rationale (Section 4.3):
      Applied after UIB Stage 2, where mid-level features at 14×14 spatial
      resolution capture edges, textures, and lesion boundaries — the ideal
      scale for spatial attention to distinguish foreground from background.

    Args:
        channels  : number of feature map channels.
        ratio     : SE-style bottleneck ratio for the channel sub-module.
        dropout   : optional dropout on the attention branch (default 0.10,
                    following PMQ-Net training hyperparameters).
    """

    def __init__(
        self,
        channels: int,
        ratio: float = 0.25,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention()
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)    # recalibrate channels
        x = self.spatial_att(x)    # recalibrate spatial positions
        return self.drop(x)


# ---------------------------------------------------------------------------
# Universal Inverted Bottleneck (UIB) — core MobileNetV4 building block
# ---------------------------------------------------------------------------

class UIBBlock(nn.Module):
    """Universal Inverted Bottleneck block.

    Generalises the MobileNetV2 inverted residual by optionally prepending
    an *extra depthwise* (ExtraDW) conv before the strided depthwise conv.
    This gives a wider effective receptive field with minimal extra params.

    Structure when use_extra_dw=True (Section 4.1):
        Input
          ↓ Pointwise expand  : (C_in -> C_in*t)   Conv1×1 + BN + ReLU6
          ↓ ExtraDW           : (hidden -> hidden)  DW3×3   + BN + ReLU6
          ↓ Strided DW        : (hidden -> hidden)  DW3×3,s + BN + ReLU6
          ↓ Pointwise project : (hidden -> C_out)   Conv1×1 + BN  [linear]
          ↕ Skip connection   : (if stride=1 and C_in==C_out)

    Args:
        in_channels   : number of input channels.
        out_channels  : number of output channels.
        expansion     : channel expansion factor t ∈ {4, 6}.
        stride        : depthwise convolution stride (1 or 2).
        use_extra_dw  : whether to add the ExtraDW layer (True in stages 2/3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        stride: int = 1,
        use_extra_dw: bool = False,
    ) -> None:
        super().__init__()
        hidden = in_channels * expansion

        # Build layers list dynamically to keep the Sequential clean
        layers: list[nn.Module] = [
            # [1] Pointwise expand
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ]

        if use_extra_dw:
            # [2] Extra depthwise conv (UIB-specific addition)
            layers += [
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]

        layers += [
            # [3] Strided depthwise conv
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # [4] Pointwise project — linear activation (no ReLU)
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.conv = nn.Sequential(*layers)

        # [5] Residual skip only when shape is preserved
        self.use_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_skip:
            out = out + x   # element-wise residual
        return out


# ---------------------------------------------------------------------------
# Superpixel Projector
# ---------------------------------------------------------------------------

class SuperpixelProjector(nn.Module):
    """Two-layer MLP projector: 96 feature channels -> E=9 encoding elements.

    Applied *per spatial position* (i.e., per superpixel) after flattening
    the 7×7 backbone output to (B, 49, 96).

    Design (Section 4.5):
        Linear(96 -> 48) + ReLU6   [first compression]
        Linear(48 -> 9)  + ReLU6   [final to E=9 elements]

    E=9 rationale:
        • 9/3 = 3 qe qubits (each U3 gate encodes 3 Euler angles)
        • Consistent with Fan et al. (2025) superpixel element count
        • 49×9 = 441 total values — richer than Wu et al.'s 16×16=256
    """

    def __init__(self, in_dim: int = 96, mid_dim: int = 48, out_dim: int = 9) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(mid_dim, out_dim),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 49, 96)  ->  out : (B, 49, 9)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Spatial Superpixel Attention (SSA) — novel component
# ---------------------------------------------------------------------------

class SpatialSuperpixelAttention(nn.Module):
    """SE-style attention over the 49 superpixel positions (Section 4.6).

    Learns WHICH patches are diagnostically relevant BEFORE quantum encoding,
    preventing the quantum circuit from wasting capacity on background regions.

    This is a novel component absent from all prior HQCNN literature.

    Mechanism:
        (B, 49, 9)
          ↓ Global average pool over element dim -> (B, 49)
          ↓ FC(49, 12) + ReLU
          ↓ FC(12, 49) + Sigmoid -> spatial importance weights (B, 49)
          ↓ Broadcast multiply  -> (B, 49, 9)
    """

    def __init__(self, n_patches: int = 49, hidden: int = 12) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_patches, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_patches),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (B, 49, 9)
        # Pool over element dimension to get per-patch scalar descriptor
        desc = x.mean(dim=-1)                    # (B, 49)
        weights = self.gate(desc)                # (B, 49)
        # Return both the re-weighted superpixels and the raw attention weights;
        # the weights are reused for attention-conditioned quantum encoding.
        return x * weights.unsqueeze(-1), weights   # (B, 49, 9), (B, 49)


# ---------------------------------------------------------------------------
# Full Classical Backbone
# ---------------------------------------------------------------------------

class ClassicalBackbone(nn.Module):
    """MobileNetV4-inspired backbone with hierarchical attention.

    Full topology (Section 4.4):
        Input (B, C_in, 28, 28)
          ↓ Stem  : Conv2d(C_in->16, 3×3, pad=1) + BN + ReLU6  -> (B,16,28,28)
          ↓ Stage1: UIB(16->24, t=4, s=1, exDW=F) + SE(24)      -> (B,24,28,28)
          ↓ Stage2: UIB(24->48, t=6, s=2, exDW=T) + CBAM(48)    -> (B,48,14,14)
          ↓ Stage3: UIB(48->96, t=6, s=2, exDW=T) + SE(96)      -> (B,96, 7, 7)
          ↓ Reshape -> (B, 49, 96)
          ↓ SuperpixelProjector                                  -> (B, 49,  9)
          ↓ SpatialSuperpixelAttention                          -> (B, 49,  9)

    Args:
        in_channels: number of image channels (1 for grey, 3 for RGB).
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()

        # --- Stem -----------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # --- Stage 1: UIB + SE (channel attention after stage 1) -----------
        # stride=1 -> preserves 28×28 spatial size
        # use_extra_dw=False (early stage — smaller receptive field sufficient)
        self.stage1 = UIBBlock(16, 24, expansion=4, stride=1, use_extra_dw=False)
        self.se1 = SEBlock(24, ratio=0.25)

        # --- Stage 2: UIB + CBAM (spatial+channel at mid resolution) -------
        # stride=2 -> 28×28 -> 14×14
        # use_extra_dw=True (UIB-specific: wider receptive field)
        self.stage2 = UIBBlock(24, 48, expansion=6, stride=2, use_extra_dw=True)
        self.cbam = CBAMBlock(48, ratio=0.25, dropout=0.10)

        # --- Stage 3: UIB + SE (channel attention again at low resolution) --
        # stride=2 -> 14×14 -> 7×7
        self.stage3 = UIBBlock(48, 96, expansion=6, stride=2, use_extra_dw=True)
        self.se3 = SEBlock(96, ratio=0.25)

        # --- Superpixel projection + attention ------------------------------
        self.projector = SuperpixelProjector(96, 48, 9)
        self.ssa = SpatialSuperpixelAttention(n_patches=49, hidden=12)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the backbone.

        Returns:
            superpixels  : (B, 49, 9) — SSA-weighted superpixel vectors,
                           ready for quantum encoding.
            z_c          : (B, 96)   — global average-pooled backbone
                           features, used for classical-quantum fusion.
            attn_weights : (B, 49)   — raw SSA per-patch importance weights,
                           used for attention-conditioned quantum encoding.
        """
        # Feature extraction
        x = self.stem(x)       # (B, 16, 28, 28)

        x = self.stage1(x)     # (B, 24, 28, 28)
        x = self.se1(x)        # channel attention

        x = self.stage2(x)     # (B, 48, 14, 14)
        x = self.cbam(x)       # spatial + channel attention

        x = self.stage3(x)     # (B, 96,  7,  7)
        x = self.se3(x)        # channel attention

        # Save global classical context for CQ fusion (before reshaping)
        z_c = x.mean(dim=[2, 3])   # (B, 96) — global average pool

        # Reshape 7×7 spatial grid -> 49 superpixel vectors
        B, C, H, W = x.shape     # C=96, H=W=7
        superpixels = x.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 96)

        # Compress each superpixel to E=9 elements
        superpixels = self.projector(superpixels)   # (B, 49, 9)

        # Apply spatial superpixel attention (novel — prioritises relevant patches)
        superpixels, attn_weights = self.ssa(superpixels)   # (B, 49, 9), (B, 49)

        return superpixels, z_c, attn_weights
