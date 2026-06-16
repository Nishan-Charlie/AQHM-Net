"""
model.py
--------
Top-level AQHM-Net model combining all sub-modules into a single nn.Module.

Full forward pass (Section 12 — Architecture Summary):

    Input (B, C_in, 28, 28)
        ↓
    ClassicalBackbone  -> superpixels (B,49,9), z_c (B,96)
        ↓
    QuantumLayer       -> z_q (B,9)
        ↓
    ClassicalQuantumFusion -> h_fused (B,64)
        ↓
    ClassificationHead -> logits (B, num_classes)

This module also provides the contrastive NT-Xent loss term for multi-modal
RGB MedMNIST datasets (Section 14.4):

    L_total = L_CE + 0.15 * L_NT-Xent(h_c, h_q)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classical_backbone import ClassicalBackbone
from .quantum_circuit import QuantumLayer, N_QUBITS, N_LAYERS
from .fusion import ClassicalQuantumFusion, ClassificationHead


# ---------------------------------------------------------------------------
# NT-Xent contrastive loss (InfoNCE variant)
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) contrastive loss.

    Used for multi-modal RGB MedMNIST experiments to align the classical
    and quantum feature representations (Section 14.4).

    Rationale: preferred over cosine similarity because it explicitly
    separates representations from DIFFERENT samples, preventing degenerate
    mode collapse where h_c and h_q collapse to the same point (PMQ-Net).

    Args:
        temperature : scaling temperature τ (default 0.07 from SimCLR).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        h_c: torch.Tensor,
        h_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss between classical and quantum projections.

        Args:
            h_c : (B, D) classical projected features (L2-normalised internally).
            h_q : (B, D) quantum projected features  (L2-normalised internally).

        Returns:
            Scalar contrastive loss.
        """
        B = h_c.shape[0]
        # L2 normalise both sets
        h_c = F.normalize(h_c, dim=-1)
        h_q = F.normalize(h_q, dim=-1)

        # Concatenate to form 2B×D feature matrix
        z = torch.cat([h_c, h_q], dim=0)     # (2B, D)

        # Full similarity matrix: (2B, 2B), scaled by temperature
        sim = (z @ z.T) / self.temperature    # (2B, 2B)

        # Remove self-similarities on the diagonal
        mask = (~torch.eye(2 * B, dtype=torch.bool, device=sim.device)).float()
        sim = sim * mask - 1e9 * (1.0 - mask)   # mask out diagonal

        # Positive pairs: for sample i (h_c), its pair is i+B (h_q) and vice versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=sim.device),   # h_c pairs -> h_q
            torch.arange(0, B, device=sim.device),        # h_q pairs -> h_c
        ])

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Full AQHM-Net Model
# ---------------------------------------------------------------------------

class AQHMNet(nn.Module):
    """Attention-Guided Hybrid Quantum MobileNet (AQHM-Net).

    Novel contributions implemented here:
        C1 — UIB Superpixel Projector (ClassicalBackbone)
        C2 — Spatial Superpixel Attention/SSA (ClassicalBackbone)
        C3 — Hierarchical SE+CBAM+SE attention backbone (ClassicalBackbone)
        C4 — Cross-paper validated: CZ gates, X-basis meas, heterogeneous
              kernels, near-zero VQC init, 3-group LR (QuantumLayer)
        C5 — First HQCNN evaluated on MedMNIST

    Args:
        in_channels      : image channels (1=grey, 3=RGB).
        num_classes      : number of output classes.
        use_contrastive  : if True, also return NT-Xent loss term (for RGB
                           MedMNIST datasets). Default False.
        contrastive_weight: λ for L_total = L_CE + λ·L_NT-Xent (default 0.15).
        n_quantum_heads  : K parallel quantum circuits over patch groups
                           (optional). K=1 (default) = original single circuit;
                           K>1 widens the quantum channel to 9K dims and keeps
                           coarse spatial structure. Ablate over {1,2,4,8}.
        attention_encoding: if True, replace the fixed SEQNN-style
                           arctan(mean(patches)) encoding with an
                           attention-conditioned TRAINABLE feature map (the SSA
                           weights pool the patches and a learned linear maps to
                           encoding angles). Default False = original behaviour.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.15,
        n_quantum_heads: int = 1,
        attention_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.n_quantum_heads = n_quantum_heads
        self.attention_encoding = attention_encoding

        # ── Sub-modules ────────────────────────────────────────────────────
        self.backbone = ClassicalBackbone(in_channels=in_channels)
        self.quantum  = QuantumLayer(
            classical_dim=96,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            n_heads=n_quantum_heads,
            attention_encoding=attention_encoding,
        )
        self.fusion   = ClassicalQuantumFusion(
            quantum_dim=self.quantum.output_dim,   # 9 * n_quantum_heads
            classical_dim=96,
            fused_dim=64,
        )
        self.head     = ClassificationHead(
            fused_dim=64,
            num_classes=num_classes,
            dropout=0.40,
        )

        if use_contrastive:
            self.nt_xent = NTXentLoss(temperature=0.07)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x : (B, C_in, 28, 28) input image batch.

        Returns:
            logits       : (B, num_classes) when use_contrastive=False.
            (logits, L_c): additionally returns the contrastive loss scalar
                           when use_contrastive=True.  Callers should then
                           compute:  L_total = L_CE(logits, y) + λ·L_c
        """
        # 1. Classical backbone: rich feature extraction + attention
        superpixels, z_c, attn_w = self.backbone(x)   # (B,49,9), (B,96), (B,49)

        # 2. Quantum layer: encode + convolve + re-upload + measure
        z_q = self.quantum(superpixels, z_c, attn_w)   # (B, 9*K)

        # 3. Classical-Quantum soft attention fusion
        h_fused = self.fusion(z_q, z_c)        # (B,64)

        # 4. Classification head -> raw logits
        logits = self.head(h_fused)             # (B, num_classes)

        if self.use_contrastive:
            # Extract projected sub-vectors for NT-Xent
            h_q = self.fusion.q_proj(z_q)      # (B, 64) — already computed,
            h_c = self.fusion.c_proj(z_c)      # reuse projections
            L_c = self.nt_xent(h_c, h_q)
            return logits, L_c

        return logits

    # ── Ablation helpers ────────────────────────────────────────────────────

    @classmethod
    def ablation_no_uib(cls, in_channels: int, num_classes: int) -> "AQHMNet":
        """A1 — Replace UIB backbone with single Conv2d (Wu et al., 2025).

        Tests: contribution of MobileNetV4-style feature extraction.
        Substitutes backbone.stem with a plain conv, bypasses stages 1-3.
        """
        import warnings
        warnings.warn(
            "ABLATION A1 (NoUIB): backbone replaced with single Conv2d; "
            "superpixel projector and SSA retained.",
            stacklevel=2,
        )
        model = cls(in_channels=in_channels, num_classes=num_classes)
        # Swap the full backbone stem+stages with a single Conv2d
        model.backbone.stage1 = nn.Identity()
        model.backbone.se1    = nn.Identity()
        model.backbone.stage2 = nn.Identity()
        model.backbone.cbam   = nn.Identity()
        model.backbone.stage3 = nn.Identity()
        model.backbone.se3    = nn.Identity()
        # Replace stem with a larger single Conv2d to produce 96-ch 7×7
        model.backbone.stem = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU6(inplace=True),
        )
        return model

    @classmethod
    def ablation_z_basis(cls, in_channels: int, num_classes: int) -> "AQHMNet":
        """A6 — Replace X-basis measurement with Z-basis (default PennyLane).

        Tests: contribution of the empirically validated X-basis measurement
        (Fan et al., 2025).  Implementation note: measurement observable swap
        is done at circuit level; this factory method is a marker for configs.
        """
        import warnings
        warnings.warn(
            "ABLATION A6 (ZBasis): model uses Z-basis PauliZ measurement. "
            "Edit quantum_circuit.py build_quantum_circuit() accordingly.",
            stacklevel=2,
        )
        return cls(in_channels=in_channels, num_classes=num_classes)
