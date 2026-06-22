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


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round a (width-scaled) channel count to the nearest multiple of `divisor`,
    never dropping below ~90% of the requested value. Standard MobileNet practice
    that keeps channel counts hardware-friendly under a width multiplier."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


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
    """Spatially-aware superpixel projector capturing local neighborhood correlations.

    Replaces the position-independent projection with a 3x3 depthwise separable conv
    over the 7x7 spatial grid before flattening. This allows spatial communication
    and local structure preservation.
    """

    def __init__(self, in_dim: int = 96, mid_dim: int = 48, out_dim: int = 9) -> None:
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU6(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Dropout2d(0.15),
            nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, in_dim, 7, 7)
        x = self.spatial_conv(x)  # (B, mid_dim, 7, 7)
        x = self.proj(x)          # (B, out_dim, 7, 7)
        # Flatten spatial dims to 49 superpixels: (B, out_dim, 49) -> (B, 49, out_dim)
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Spatial Superpixel Attention (SSA) — novel component
# ---------------------------------------------------------------------------

class SpatialSuperpixelAttention(nn.Module):
    """Spatially-aware superpixel attention utilizing 2D convolutions.

    Rather than treating the 7x7 grid as a flat 49-D vector, this module
    operates directly on the 2D spatial topology to learn local attention weight
    distributions using 2D conv layers.
    """

    def __init__(self, n_patches: int = 49, hidden: int = 12) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x : (B, 49, 9)
        B, P, E = x.shape
        H = W = 7
        # Pool over element dimension to get spatial descriptor: (B, 49) -> (B, 1, 7, 7)
        desc = x.mean(dim=-1).view(B, 1, H, W)
        
        # Apply 2D spatial convolutions
        weights = self.conv(desc).view(B, P)     # (B, 49)
        
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
        width_mult : channel width multiplier (1.0 = base; >1 scales all
                     backbone channels). Used by the small/medium/large presets.
        depth      : number of UIB blocks per stage (1 = base; >1 stacks extra
                     residual same-dimension blocks for more depth/capacity).
    """

    def __init__(
        self,
        in_channels: int = 1,
        width_mult: float = 1.0,
        depth: int = 1,
        img_size: int = 28,
    ) -> None:
        super().__init__()

        def ch(c: int) -> int:
            return _make_divisible(c * width_mult)

        c_stem, c1, c2, c3 = ch(16), ch(24), ch(48), ch(96)
        self.out_channels = c3   # consumed by the projector / quantum / fusion

        # Determine progressive strides based on resolution
        if img_size >= 224:
            stem_stride = 2      # 224 -> 112
            stage1_stride = 2    # 112 -> 56
            stage2_stride = 2    # 56 -> 28
            stage3_stride = 2    # 28 -> 14
            self.use_downsample_block = True
        elif img_size >= 128:
            stem_stride = 2      # 128 -> 64
            stage1_stride = 2    # 64 -> 32
            stage2_stride = 2    # 32 -> 16
            stage3_stride = 2    # 16 -> 8
            self.use_downsample_block = False
        elif img_size >= 64:
            stem_stride = 2      # 64 -> 32
            stage1_stride = 1    # 32 -> 32
            stage2_stride = 2    # 32 -> 16
            stage3_stride = 2    # 16 -> 8
            self.use_downsample_block = False
        else:
            stem_stride = 1      # 28 -> 28
            stage1_stride = 1    # 28 -> 28
            stage2_stride = 2    # 28 -> 14
            stage3_stride = 2    # 14 -> 7
            self.use_downsample_block = False

        # --- Stem -----------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_stem, kernel_size=3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU6(inplace=True),
        )

        # --- Stage 1: UIB(+depth) + SE -------------------------------------
        self.stage1 = self._make_stage(c_stem, c1, expansion=4, stride=stage1_stride,
                                       extra_dw=False, n=depth)
        self.se1 = SEBlock(c1, ratio=0.25)

        # --- Stage 2: UIB(+depth) + CBAM (mid resolution) ------------------
        self.stage2 = self._make_stage(c1, c2, expansion=6, stride=stage2_stride,
                                       extra_dw=True, n=depth)
        self.cbam = CBAMBlock(c2, ratio=0.25, dropout=0.10)

        # --- Stage 3: UIB(+depth) + SE -------------------------------------
        self.stage3 = self._make_stage(c2, c3, expansion=6, stride=stage3_stride,
                                       extra_dw=True, n=depth)
        self.se3 = SEBlock(c3, ratio=0.25)

        # Learnable stride-2 transition block to downsample from 14x14 to 7x7
        if self.use_downsample_block:
            self.downsample_conv = nn.Sequential(
                nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1, groups=c3, bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU6(inplace=True),
                nn.Conv2d(c3, c3, kernel_size=1, bias=False),
                nn.BatchNorm2d(c3),
            )
        else:
            self.downsample_conv = nn.Identity()

        # Superpixel projector / SSA hidden also scale modestly with width.
        self._proj_mid = ch(48)

        # --- Resolution-adaptive pooling ------------------------------------
        self.feat_pool = nn.AdaptiveAvgPool2d((7, 7))

        # --- Superpixel projection + attention ------------------------------
        # Projector input = scaled backbone output channels (self.out_channels).
        self.projector = SuperpixelProjector(self.out_channels, self._proj_mid, 9)
        self.ssa = SpatialSuperpixelAttention(n_patches=49, hidden=12)

    @staticmethod
    def _make_stage(in_c: int, out_c: int, expansion: int, stride: int,
                    extra_dw: bool, n: int) -> nn.Sequential:
        """A stage = one shape-changing UIB block followed by (n-1) residual
        same-dimension UIB blocks (depth multiplier)."""
        blocks = [UIBBlock(in_c, out_c, expansion, stride, extra_dw)]
        for _ in range(max(0, n - 1)):
            blocks.append(UIBBlock(out_c, out_c, expansion, 1, extra_dw))
        return nn.Sequential(*blocks)

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
        x = self.stem(x)

        x = self.stage1(x)     # channel attention
        x = self.se1(x)

        x = self.stage2(x)     # spatial + channel attention
        x = self.cbam(x)

        x = self.stage3(x)     # channel attention
        x = self.se3(x)

        # Learnable spatial downsampling if enabled
        if self.use_downsample_block:
            x = self.downsample_conv(x)

        # Resolution-adaptive fallback average pool to 7×7 grid
        x = self.feat_pool(x)  # (B, 96, 7, 7)

        # Save global classical context for CQ fusion (before flattening)
        z_c = x.mean(dim=[2, 3])   # (B, 96) — global average pool

        # Compress spatial grid of shape (B, C, 7, 7) to E=9 elements using spatial convolutions
        superpixels = self.projector(x)             # (B, 49, 9)

        # Apply spatial superpixel attention (novel — prioritises relevant patches)
        superpixels, attn_weights = self.ssa(superpixels)   # (B, 49, 9), (B, 49)

        return superpixels, z_c, attn_weights
