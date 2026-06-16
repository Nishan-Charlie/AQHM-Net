"""
fusion.py
---------
Classical-Quantum soft attention fusion and classification head for AQHM-Net.

Section 10 — Classical-Quantum Fusion:
    Projects quantum (9-dim) and classical (96-dim) features into a shared
    64-dim space, then combines them via a LEARNABLE soft attention gate:

        h_fused = α · h_q + (1 − α) · h_c

    where α = Sigmoid(Linear(128, 1)([h_q; h_c])) ∈ (0, 1).

    This allows dynamic, per-sample balancing between quantum and classical
    contributions (PMQ-Net, 2025):
        • Simple tasks (MNIST digits): gate may favour classical features
        • Ambiguous medical images:    gate may weight quantum correlations

Section 10 — Classification Head:
    h_fused (64) -> Linear(64->32) + ReLU + Dropout(0.40) -> Linear(32->C)

    Dropout 0.40 follows PMQ-Net (2025), effective for small quantum feature
    vectors to prevent overfitting.
"""

import torch
import torch.nn as nn


class ClassicalQuantumFusion(nn.Module):
    """Soft attention gate to fuse quantum and classical feature vectors.

    Args:
        quantum_dim   : dimensionality of the quantum measurement vector (9).
        classical_dim : dimensionality of the classical backbone context (96).
        fused_dim     : shared projected dimension (64).
    """

    def __init__(
        self,
        quantum_dim: int = 9,
        classical_dim: int = 96,
        fused_dim: int = 64,
    ) -> None:
        super().__init__()

        # Project quantum (9) -> shared space (64)
        self.q_proj = nn.Linear(quantum_dim, fused_dim)
        # Project classical (96) -> shared space (64)
        self.c_proj = nn.Linear(classical_dim, fused_dim)

        # Soft attention gate: concatenated (128) -> 32 -> scalar α ∈ (0,1)
        # Deeper gate learns richer quantum-classical interaction patterns
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_q: torch.Tensor,
        z_c: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse quantum and classical vectors.

        Args:
            z_q : (B, 9)  quantum feature vector from the VQC.
            z_c : (B, 96) classical backbone global average-pool vector.

        Returns:
            h_fused : (B, 64) fused representation.
        """
        h_q = self.q_proj(z_q)      # (B, 64)
        h_c = self.c_proj(z_c)      # (B, 64)

        # Attention weight α — depends on both representations
        alpha = self.gate(torch.cat([h_q, h_c], dim=-1))  # (B, 1)

        # Soft interpolation between quantum and classical
        h_fused = alpha * h_q + (1.0 - alpha) * h_c       # (B, 64)
        return h_fused


class ClassificationHead(nn.Module):
    """Two-layer MLP classification head.

    Architecture (Section 11):
        Linear(64 -> 32) + ReLU + Dropout(0.40)
        Linear(32 -> num_classes)
        -> class logits (softmax applied during loss/inference)

    Args:
        fused_dim   : input dimension from the fusion module (64).
        num_classes : number of output classes.
        dropout     : dropout rate (0.40 per PMQ-Net, 2025).
    """

    def __init__(
        self,
        fused_dim: int = 64,
        num_classes: int = 10,
        dropout: float = 0.40,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            # NOTE: no softmax here — nn.CrossEntropyLoss expects raw logits.
            #       Apply F.softmax() only during inference for probabilities.
        )

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_fused : (B, 64) fused feature vector.
        Returns:
            logits  : (B, num_classes) unnormalised class scores.
        """
        return self.head(h_fused)
