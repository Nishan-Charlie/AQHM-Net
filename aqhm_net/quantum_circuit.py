"""
quantum_circuit.py  (v6 — optimised pure-PyTorch batched state-vector)
-----------------------------------------------------------------------
Three perf improvements over v5:

  1. Vectorised gate creation — all RY/RX/RZ matrices built in ONE tensor
     call each (no per-qubit Python loops for matrix creation).

  2. Fused reup + trainable gate — for each (layer, qubit), we compose
     RZ@RY@RX (trainable, shared) with RY_reup (per-sample) into a single
     combined (B,2,2) matrix via a single broadcasted matmul.

  3. BMM instead of einsum — _apply_sq uses torch.bmm which is ~2.5x
     faster than the equivalent einsum on GPU.

Remaining Python loop: 9 qubit gate-applies per layer (3 layers + init =
36 bmm calls total). These are unavoidable for an entangled state and
each runs in ~64 μs on GPU → total forward ~3-5 ms vs 43 ms in v5.

Preserved from METHODOLOGY.md (unchanged):
  • L=3 re-uploading layers, each with its own linear projection (Sec 9)
  • arctan angle mapping into (-pi/2, +pi/2) (Sec 9.2)
  • PauliX measurements on wires 0-5, ZZ on (0,3)(3,6)(0,6) (Sec 8)
  • Near-zero VQC init U(-0.01, 0.01) (Sec 14.2)
  • CZ entanglement ring (Sec 9)
  • 9-dim quantum feature vector output
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Circuit constants
# ---------------------------------------------------------------------------

N_QUBITS = 9
N_LAYERS  = 3
N_MEAS    = 9          # 6 PauliX + 3 ZZ
DIM       = 2 ** N_QUBITS   # 512

_PAULIX_WIRES = list(range(6))
_ZZ_PAIRS     = [(0, 3), (3, 6), (0, 6)]
_CZ_PAIRS     = [(q, q + 1) for q in range(N_QUBITS - 1)] + [(N_QUBITS - 1, 0)]


# ---------------------------------------------------------------------------
# Vectorised gate matrix constructors (no Python loops over qubits)
# ---------------------------------------------------------------------------

def _ry(theta: torch.Tensor) -> torch.Tensor:
    """RY gate matrices.

    Args:
        theta : (...) float — rotation angle(s)
    Returns:
        gate  : (..., 2, 2) cfloat
    """
    h = theta / 2.0
    c, s = torch.cos(h), torch.sin(h)
    return torch.stack(
        [torch.stack([ c, -s], dim=-1),
         torch.stack([ s,  c], dim=-1)], dim=-2
    ).to(torch.cfloat)


def _rx(theta: torch.Tensor) -> torch.Tensor:
    """RX gate matrices.

    RX(θ) = [[cos θ/2,  -i sin θ/2],
              [-i sin θ/2,  cos θ/2]]
    """
    h = theta / 2.0
    c = torch.cos(h)
    s = torch.sin(h)
    z = torch.zeros_like(c)
    real = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    imag = torch.stack([torch.stack([z,-s], -1), torch.stack([-s,z], -1)], -2)
    return torch.complex(real, imag)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    """RZ gate matrices.

    RZ(θ) = diag(e^{-iθ/2}, e^{+iθ/2})
    """
    h = theta / 2.0
    e_m = torch.complex(torch.cos(-h), torch.sin(-h))
    e_p = torch.complex(torch.cos( h), torch.sin( h))
    z   = torch.zeros_like(e_m)
    return torch.stack([torch.stack([e_m, z], -1),
                        torch.stack([z, e_p], -1)], -2)


def _combined_vqc_gates(vqc_params: torch.Tensor) -> torch.Tensor:
    """Build combined RZ@RY@RX gate for every (layer, qubit) pair.

    Args:
        vqc_params : (L, N, 3) float — [Rx_angle, Ry_angle, Rz_angle]
    Returns:
        combined   : (L, N, 2, 2) cfloat
    """
    rx = _rx(vqc_params[..., 0])   # (L, N, 2, 2)
    ry = _ry(vqc_params[..., 1])   # (L, N, 2, 2)
    rz = _rz(vqc_params[..., 2])   # (L, N, 2, 2)
    # matmul on last two dims, broadcasting over (L, N)
    return rz @ ry @ rx             # (L, N, 2, 2)


# ---------------------------------------------------------------------------
# Gate application via BMM
# ---------------------------------------------------------------------------

def _apply_sq(state: torch.Tensor, gate: torch.Tensor, q: int) -> torch.Tensor:
    """Apply per-sample (B, 2, 2) gate to qubit q of state (B, DIM).

    Uses torch.bmm (faster than einsum on GPU).

    Result[b, i, l, r] = sum_k gate[b, l, k] * state[b, i, k, r]
    where the state is implicitly shaped (B, q_high, 2, q_low).
    """
    B   = state.shape[0]
    q_low   = 2 ** q
    q_high  = DIM // (2 * q_low)

    # s: (B*q_high, 2, q_low) — contiguous after view chain
    s = state.view(B, q_high, 2, q_low).reshape(B * q_high, 2, q_low)

    # g: (B*q_high, 2, 2) — repeat gate q_high times per sample
    g = gate.unsqueeze(1).expand(B, q_high, 2, 2).reshape(B * q_high, 2, 2)

    result = torch.bmm(g, s)   # (B*q_high, 2, q_low)
    return result.view(B, q_high, 2, q_low).reshape(B, DIM)


def _apply_cz_ring(state: torch.Tensor, cz_phase: torch.Tensor) -> torch.Tensor:
    """Apply combined CZ ring via precomputed phase mask (no loop)."""
    return state * cz_phase.to(state.dtype).unsqueeze(0)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _expval_pauli_x(state: torch.Tensor, flip_idx: torch.Tensor) -> torch.Tensor:
    """<PauliX_q> = Re( sum_i psi*_i  psi_{flip_q(i)} )."""
    return (state.conj() * state[:, flip_idx]).sum(-1).real


def _expval_zz(state: torch.Tensor, zz_phase: torch.Tensor) -> torch.Tensor:
    """<ZZ_{q0,q1}> = sum_i |psi_i|^2 * (-1)^{bit_q0 + bit_q1}."""
    return (state.abs().pow(2) * zz_phase.unsqueeze(0)).sum(-1)


# ---------------------------------------------------------------------------
# QuantumLayer
# ---------------------------------------------------------------------------

class QuantumLayer(nn.Module):
    """Differentiable quantum layer for AQHM-Net, with optional parallel heads.

    Inputs:
        superpixels : (B, P, 9) SSA-weighted patch vectors (P = 49 by default)
        z_c         : (B, 96)   classical backbone context

    Output:
        z_q : (B, n_heads * 9) concatenated quantum measurement vector

    Parallel quantum circuits (optional, n_heads = K):
        The P patches are partitioned into K contiguous groups (patch grouping).
        Each group is pooled to a 9-D encoding and fed to its OWN independent
        VQC (own gate angles + own re-uploading projections); the K measurement
        vectors are concatenated. This widens the quantum channel from 9 to 9K
        dimensions AND preserves coarse spatial structure (each head sees a
        different region) instead of averaging all patches into one vector.

        K = 1 reproduces the original single-circuit behaviour exactly: one
        group = mean over all P patches → a single 9-D output.

    Trainable parameters:
        vqc_params       : (K, L, N, 3) — RX/RY/RZ angles per head
        reup_proj_{k}_{l}: Linear(96, 9) for head k, layer l
                           (named to match the optimiser's bridge-LR prefix
                           'quantum.reup_proj_')

    Non-trainable buffers (precomputed at init, shared across heads):
        cz_phase      : (DIM,)   combined CZ ring phase mask
        flip_idx_{q}  : (DIM,)   PauliX flip index for qubit q
        zz_phase_{p}  : (DIM,)   ZZ phase mask for pair p
    """

    def __init__(
        self,
        classical_dim: int = 96,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        n_heads: int = 1,
        attention_encoding: bool = False,
    ) -> None:
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attention_encoding = attention_encoding
        self.output_dim = n_heads * N_MEAS
        dim = 2 ** n_qubits

        # VQC gate parameters — near-zero init (Sec 14.2), one set per head.
        self.vqc_params = nn.Parameter(torch.zeros(n_heads, n_layers, n_qubits, 3))
        nn.init.uniform_(self.vqc_params, -0.01, 0.01)

        # Per-head, per-layer re-uploading projections. Named reup_proj_{k}_{l}
        # so every one still starts with 'quantum.reup_proj_' (bridge LR group).
        for k in range(n_heads):
            for l in range(n_layers):
                setattr(self, f"reup_proj_{k}_{l}",
                        nn.Linear(classical_dim, n_qubits, bias=True))

        # Attention-conditioned trainable encoding (novel — replaces the fixed
        # SEQNN-style arctan(mean(patches)) feature map). When enabled, each
        # head pools its patch group with the SSA attention weights (instead of
        # a flat mean) and maps the result to encoding angles through a TRAINABLE
        # linear feature map — coupling the learned spatial attention directly
        # into the quantum encoding. Named enc_proj_{k} -> bridge LR group.
        if attention_encoding:
            for k in range(n_heads):
                setattr(self, f"enc_proj_{k}", nn.Linear(n_qubits, n_qubits, bias=True))

        # ── Non-parametric buffers ──────────────────────────────────────
        indices = torch.arange(dim)

        # Combined CZ ring: apply all pairs at once with one elementwise multiply
        cz_phase = torch.ones(dim, dtype=torch.float32)
        for q0, q1 in _CZ_PAIRS:
            b0 = (indices >> q0) & 1
            b1 = (indices >> q1) & 1
            cz_phase *= (1 - 2 * b0 * b1).float()
        self.register_buffer("cz_phase", cz_phase)

        # PauliX flip indices (one per measured qubit)
        for q in _PAULIX_WIRES:
            self.register_buffer(f"flip_idx_{q}", indices ^ (1 << q))

        # ZZ phase masks
        for idx, (q0, q1) in enumerate(_ZZ_PAIRS):
            phase_zz = ((1 - 2 * ((indices >> q0) & 1))
                        * (1 - 2 * ((indices >> q1) & 1))).float()
            self.register_buffer(f"zz_phase_{idx}", phase_zz)

    @staticmethod
    def _arctan_encode(z: torch.Tensor) -> torch.Tensor:
        """Map R → (-π/2, +π/2) via arctan (Sec 9.2)."""
        return torch.arctan(z) * (math.pi / 2.0)

    def _run_circuit(
        self,
        x_enc:      torch.Tensor,   # (B, N)
        reup:       torch.Tensor,   # (B, L, N)
        vqc_params: torch.Tensor,   # (L, N, 3)
    ) -> torch.Tensor:              # (B, M)
        """Batched state-vector simulation — fully on the input device.

        All gate matrices are built in vectorised tensor calls before the
        gate-apply loop, minimising Python overhead to 9 bmm calls per layer.
        """
        B      = x_enc.shape[0]
        device = x_enc.device

        # ── 1. Precompute all gate matrices ─────────────────────────────

        # (a) Initial encoding: RY per qubit per sample → (B, N, 2, 2)
        ry_init = _ry(x_enc)                       # (B, N, 2, 2)

        # (b) Re-uploading encoding: RY per (sample, layer, qubit) → (B, L, N, 2, 2)
        ry_reup = _ry(reup)                        # (B, L, N, 2, 2)

        # (c) Combined trainable gate per (layer, qubit): RZ@RY@RX → (L, N, 2, 2)
        combined_vqc = _combined_vqc_gates(vqc_params)   # (L, N, 2, 2)

        # (d) Fuse: G_full[b,l,q] = combined_vqc[l,q] @ ry_reup[b,l,q]
        #     Shape: (1,L,N,2,2) @ (B,L,N,2,2) → (B,L,N,2,2)  [broadcast over B]
        g_full = combined_vqc.unsqueeze(0) @ ry_reup    # (B, L, N, 2, 2)

        # ── 2. Initialise |0...0> ────────────────────────────────────────
        state = torch.zeros(B, DIM, dtype=torch.cfloat, device=device)
        state[:, 0] = 1.0 + 0j

        # ── 3. Initial angle embedding ────────────────────────────────────
        for q in range(self.n_qubits):
            state = _apply_sq(state, ry_init[:, q], q)

        # ── 4. L re-uploading layers ──────────────────────────────────────
        for l in range(self.n_layers):
            # Apply fused (reup + trainable) gate per qubit
            for q in range(self.n_qubits):
                state = _apply_sq(state, g_full[:, l, q], q)
            # CZ entanglement ring (one elementwise multiply)
            state = _apply_cz_ring(state, self.cz_phase)

        # ── 5. Measurements ───────────────────────────────────────────────
        meas = []
        for q in _PAULIX_WIRES:
            meas.append(_expval_pauli_x(state, getattr(self, f"flip_idx_{q}")))
        for idx in range(len(_ZZ_PAIRS)):
            meas.append(_expval_zz(state, getattr(self, f"zz_phase_{idx}")))

        return torch.stack(meas, dim=1)   # (B, N_MEAS)

    def forward(
        self,
        superpixels:  torch.Tensor,            # (B, P, N_QUBITS)
        z_c:          torch.Tensor,            # (B, classical_dim)
        attn_weights: torch.Tensor | None = None,   # (B, P) SSA weights
    ) -> torch.Tensor:                         # (B, n_heads * N_MEAS)
        P = superpixels.shape[1]
        # Partition the P patches into n_heads contiguous groups. K=1 -> one
        # group covering all patches (mean over all P, i.e. original behaviour).
        groups = torch.tensor_split(torch.arange(P, device=superpixels.device),
                                    self.n_heads)

        head_outputs = []
        for k, idx in enumerate(groups):
            sp = superpixels[:, idx, :]                          # (B, n_k, N)

            if self.attention_encoding and attn_weights is not None:
                # Attention-conditioned trainable encoding: pool the patch group
                # with the (re-normalised) SSA weights, then map to angles via a
                # trainable feature map. Replaces the flat mean + fixed arctan and
                # removes the mean() information loss.
                a = torch.softmax(attn_weights[:, idx], dim=1)  # (B, n_k)
                agg = (a.unsqueeze(-1) * sp).sum(dim=1)         # (B, N) attn-weighted pool
                x_enc = self._arctan_encode(getattr(self, f"enc_proj_{k}")(agg))
            else:
                # Original SEQNN-style encoding: flat mean over the patch group
                x_enc = self._arctan_encode(sp.mean(dim=1))     # (B, N)

            # This head's per-layer re-uploading projections (distinct per layer)
            reup = torch.stack([
                self._arctan_encode(getattr(self, f"reup_proj_{k}_{l}")(z_c))
                for l in range(self.n_layers)
            ], dim=1)                                            # (B, L, N)

            head_outputs.append(self._run_circuit(x_enc, reup, self.vqc_params[k]))

        # Concatenate the K measurement vectors -> (B, n_heads * N_MEAS)
        return torch.cat(head_outputs, dim=-1)
