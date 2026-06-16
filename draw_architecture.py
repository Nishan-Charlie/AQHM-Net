"""
draw_architecture.py
--------------------
Generates the AQHM-Net architecture diagram for the paper.
Saves: results/plots/aqhm_net_architecture.png  (300 DPI)
       results/plots/aqhm_net_architecture.pdf
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Okabe-Ito palette ─────────────────────────────────────────────────────────
C_CLASSICAL = "#0072B2"   # blue
C_QUANTUM   = "#E69F00"   # amber
C_ATTN      = "#009E73"   # green
C_FUSION    = "#CC79A7"   # pink
C_HEAD      = "#56B4E9"   # sky-blue
C_INPUT     = "#555555"   # dark grey
C_NOVEL     = "#D55E00"   # vermilion (novel contribution badges)
C_BG        = "#F8F8F8"

FIG_W, FIG_H = 14.0, 20.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="white")
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, color, label, sublabel="", alpha=0.92,
         radius=0.25, fontsize=9, label_color="white", novel=False):
    """Draw a rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.2, edgecolor=color,
        facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=label_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel,
                ha="center", va="center", fontsize=7.5,
                color=label_color, alpha=0.9, zorder=4)
    if novel:
        star_x = x + w / 2 - 0.15
        star_y = y + h / 2 - 0.15
        ax.text(star_x, star_y, "★", ha="center", va="center",
                fontsize=9, color=C_NOVEL, fontweight="bold", zorder=5)


def section_box(ax, x, y, w, h, color, title, alpha=0.08):
    """Light background rectangle for a section group."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.3",
        linewidth=1.4, edgecolor=color, linestyle="--",
        facecolor=color, alpha=alpha, zorder=1,
    )
    ax.add_patch(box)
    ax.text(x - w / 2 + 0.18, y + h / 2 - 0.22, title,
            ha="left", va="top", fontsize=7.5,
            color=color, fontweight="bold", alpha=0.85, zorder=2)


def arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=5)


def dim_label(ax, x, y, text, color="#888888"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=7, color=color, style="italic",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1), zorder=6)

# ── Layout constants ──────────────────────────────────────────────────────────
CX   = FIG_W / 2          # centre x
BW   = 9.0                # box width (main)
QW   = 8.2                # quantum box width
BH   = 0.62               # standard box height
GAP  = 0.28               # gap between boxes

# Y positions (top-to-bottom)
y_input     = 19.3
y_stem      = 18.4
y_uib1      = 17.55
y_uib2      = 16.65
y_uib3      = 15.75

y_proj      = 14.65
y_ssa       = 13.75

y_qenc      = 12.50
y_qconv     = 11.55
y_hadamard  = 10.60
y_measure   = 9.65

y_fusion    = 8.45
y_head1     = 7.55
y_head2     = 6.70
y_output    = 5.80

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(CX, 19.82, "AQHM-Net Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")
ax.text(CX, 19.52, "Attention-Guided Hybrid Quantum MobileNet",
        ha="center", va="center", fontsize=9, color="#555555")

# ── INPUT ─────────────────────────────────────────────────────────────────────
rbox(ax, CX, y_input, 5.5, BH, C_INPUT, "Input Image", "(B, C_in, 28×28)", fontsize=9)
dim_label(ax, CX + 3.2, y_input, "C_in ∈ {1, 3}")

# ── CLASSICAL BACKBONE section ────────────────────────────────────────────────
section_box(ax, CX, (y_stem + y_uib3) / 2, BW + 1.0,
            y_stem - y_uib3 + BH + 0.55, C_CLASSICAL,
            "Classical Backbone  —  UIB + Hierarchical Attention")

arrow(ax, CX, y_input - BH / 2, CX, y_stem + BH / 2)

rbox(ax, CX, y_stem, BW, BH, C_CLASSICAL,
     "Stem  Conv2d(C_in→16, 3×3) + BN + ReLU6",
     "(B, 16, 28×28)", fontsize=8.5)

arrow(ax, CX, y_stem - BH / 2, CX, y_uib1 + BH / 2)
rbox(ax, CX, y_uib1, BW, BH, C_CLASSICAL,
     "Stage 1:  UIB(16→24, t=4, s=1)  +  SE(r=0.25)",
     "(B, 24, 28×28)", fontsize=8.5)

arrow(ax, CX, y_uib1 - BH / 2, CX, y_uib2 + BH / 2)
rbox(ax, CX, y_uib2, BW, BH, "#007B55",   # darker green for CBAM stage
     "Stage 2:  UIB(24→48, t=6, s=2)  +  CBAM(r=0.25)",
     "(B, 48, 14×14)", fontsize=8.5)

arrow(ax, CX, y_uib2 - BH / 2, CX, y_uib3 + BH / 2)
rbox(ax, CX, y_uib3, BW, BH, C_CLASSICAL,
     "Stage 3:  UIB(48→96, t=6, s=2)  +  SE(r=0.25)",
     "(B, 96, 7×7)", fontsize=8.5)

# SE / CBAM small labels on right
for yt, lbl in [(y_uib1, "SE"), (y_uib2, "CBAM\n+SE"), (y_uib3, "SE")]:
    ax.text(CX + BW / 2 + 0.38, yt, lbl,
            ha="center", va="center", fontsize=7, color=C_ATTN,
            fontweight="bold",
            bbox=dict(fc=C_ATTN, ec="none", alpha=0.15, pad=2, boxstyle="round"))

# ── SUPERPIXEL PROJECTION section ─────────────────────────────────────────────
section_box(ax, CX, (y_proj + y_ssa) / 2, BW + 1.0,
            y_proj - y_ssa + BH + 0.55, C_ATTN,
            "Superpixel Projection  +  SSA  ★ novel")

arrow(ax, CX, y_uib3 - BH / 2, CX, y_proj + BH / 2)
rbox(ax, CX, y_proj, BW, BH, C_ATTN,
     "UIB Projector:  Reshape → Linear(96→48) → Linear(48→9) + ReLU6",
     "(B, 49, 9)  [49 superpixels × 9 elements]",
     fontsize=8.5, novel=True)

arrow(ax, CX, y_proj - BH / 2, CX, y_ssa + BH / 2)
rbox(ax, CX, y_ssa, BW, BH, C_ATTN,
     "Spatial Superpixel Attention (SSA):  GAP → FC(49→12) → FC(12→49) + Sigmoid",
     "Attention-weighted superpixels  (B, 49, 9)",
     fontsize=8.5, novel=True)

# ── SPLIT: z_c branch (dashed, left) ─────────────────────────────────────────
# z_c comes from backbone GlobalAvgPool (shown as side branch)
zc_x = CX - 4.5
zc_y = (y_ssa + y_fusion) / 2

ax.annotate("", xy=(zc_x, y_fusion + 0.22), xytext=(CX - BW / 2, y_uib3),
            arrowprops=dict(arrowstyle="-|>", color=C_CLASSICAL,
                            lw=1.3, linestyle="dashed", mutation_scale=10),
            zorder=4)
rbox(ax, zc_x, zc_y, 2.8, 0.52, C_CLASSICAL,
     "z_c  =  GAP(Stage 3)", "(B, 96)", fontsize=8, alpha=0.85)
ax.text(zc_x, zc_y - 0.55, "classical\ncontext", ha="center", va="top",
        fontsize=7, color=C_CLASSICAL, style="italic")

# ── QUANTUM CIRCUIT section ───────────────────────────────────────────────────
q_sec_cy = (y_qenc + y_measure) / 2
q_sec_h  = y_qenc - y_measure + BH + 0.65
section_box(ax, CX + 0.3, q_sec_cy, QW + 1.0, q_sec_h, C_QUANTUM,
            "Variational Quantum Circuit  (9 data + 2 ancilla = 11 qubits, L=3 re-upload layers)")

arrow(ax, CX, y_ssa - BH / 2, CX, y_qenc + BH / 2)

# Quantum Encoding
rbox(ax, CX + 0.3, y_qenc, QW, BH, C_QUANTUM,
     "Quantum Encoding:  ql (6 qubits, H + CU3 per superpixel)  +  qe (3 qubits, U3 + CZ all-to-all)",
     "Data re-uploading per layer:  arctan(Linear_l(z_c)) → Ry angles",
     fontsize=8, label_color="white")

# Kernel boxes inside quantum section
kern_y = y_qconv
kern_w = 1.65
kern_gap = 0.18
kern_colors = ["#B07D00", "#9E6B00", "#7A5300", "#5C3D00"]  # amber shades
kern_labels = [
    ("Kernel A", "U2\n2-qubit"),
    ("Kernel B", "U3_CB\n3-qubit\nCircle"),
    ("Kernel C", "U4_NN\n4-qubit\nNN"),
    ("Kernel D", "U4_AA\n4-qubit\nAll-All"),
]
total_kern_w = 4 * kern_w + 3 * kern_gap
kern_start_x = CX + 0.3 - total_kern_w / 2 + kern_w / 2

arrow(ax, CX + 0.3, y_qenc - BH / 2, CX + 0.3, kern_y + 0.55)
ax.text(CX + 0.3, kern_y + 0.70,
        "Quantum Convolutional Layer  (4 heterogeneous kernels in ancilla superposition)",
        ha="center", va="center", fontsize=8, color="#5C3D00", fontweight="bold")

for ki, (klbl, ksub) in enumerate(kern_labels):
    kx = kern_start_x + ki * (kern_w + kern_gap)
    rbox(ax, kx, kern_y, kern_w, 0.90, C_QUANTUM,
         klbl, ksub, fontsize=7.5, radius=0.2, alpha=0.88)
    if ki < 3:
        ax.text(kx + kern_w / 2 + kern_gap / 2, kern_y,
                "⊕", ha="center", va="center", fontsize=10, color=C_QUANTUM)

# Hadamard integration
arrow(ax, CX + 0.3, kern_y - 0.48, CX + 0.3, y_hadamard + BH / 2)
rbox(ax, CX + 0.3, y_hadamard, QW, BH, C_QUANTUM,
     "Hadamard Integration:  H⊗H on ancilla → post-select |00⟩ → channel fusion",
     "SSA-weighted amplitude superposition  (constructive interference)",
     fontsize=8, label_color="white")

# Measurement
arrow(ax, CX + 0.3, y_hadamard - BH / 2, CX + 0.3, y_measure + BH / 2)
rbox(ax, CX + 0.3, y_measure, QW, BH, C_QUANTUM,
     "Measurement:  6 × ⟨PauliX⟩  +  3 × ⟨ZZ⟩ correlators",
     "z_q  (B, 9-dim quantum feature vector)",
     fontsize=8, label_color="white")

# ── FUSION ────────────────────────────────────────────────────────────────────
# Arrow from quantum measurement
arrow(ax, CX + 0.3, y_measure - BH / 2, CX, y_fusion + BH / 2 + 0.06,
      color=C_QUANTUM)
# Arrow from z_c
arrow(ax, zc_x + 1.4, zc_y, CX - BW / 2 + 0.05, y_fusion,
      color=C_CLASSICAL)

section_box(ax, CX, (y_fusion + y_head1) / 2, BW + 1.0,
            y_fusion - y_head1 + BH + 0.60, C_FUSION,
            "Classical-Quantum Fusion  ★ novel")

rbox(ax, CX, y_fusion, BW, BH, C_FUSION,
     "Soft Attention Gate:  h_q = Linear(9→64),  h_c = Linear(96→64)",
     "α = Sigmoid(Linear(128→1)[h_q ‖ h_c])    h_fused = α·h_q + (1−α)·h_c   (B, 64)",
     fontsize=8, novel=True)

# alpha annotation
ax.text(CX + BW / 2 - 0.05, y_fusion + 0.32, "α ∈ (0,1)\nper sample",
        ha="right", va="center", fontsize=6.5, color=C_FUSION, style="italic")

# ── CLASSIFICATION HEAD ───────────────────────────────────────────────────────
arrow(ax, CX, y_fusion - BH / 2, CX, y_head1 + BH / 2)
rbox(ax, CX, y_head1, BW, BH, C_HEAD,
     "Classification Head:  Linear(64→32) + ReLU + Dropout(0.40)",
     "(B, 32)", fontsize=8.5, label_color="white")

arrow(ax, CX, y_head1 - BH / 2, CX, y_head2 + BH / 2)
rbox(ax, CX, y_head2, BW, BH, C_HEAD,
     "Linear(32 → num_classes)  +  Softmax",
     "Logits  (B, K)   K ∈ {2, 7, 9, 10, 100}", fontsize=8.5, label_color="white")

arrow(ax, CX, y_head2 - BH / 2, CX, y_output + BH / 2)
rbox(ax, CX, y_output, 5.0, BH, C_INPUT,
     "Class Predictions  (B, K)", "", fontsize=9)

# ── LOSS annotation (right side) ─────────────────────────────────────────────
loss_x = CX + BW / 2 + 1.0
loss_y = (y_fusion + y_output) / 2
ax.text(loss_x, loss_y + 0.55, "Training Loss:", ha="left", va="center",
        fontsize=8, color="#333333", fontweight="bold")
ax.text(loss_x, loss_y + 0.18, "L = FocalLoss(γ=2)", ha="left", va="center",
        fontsize=7.5, color="#333333")
ax.text(loss_x, loss_y - 0.15, "  [medical datasets]", ha="left", va="center",
        fontsize=7, color="#888888", style="italic")
ax.text(loss_x, loss_y - 0.45, "L = CE  [MNIST/CIFAR]", ha="left", va="center",
        fontsize=7.5, color="#333333")

# Contrastive loss (PathMNIST only)
ax.text(loss_x, loss_y - 0.82, "+ 0.15 · L_NT-Xent", ha="left", va="center",
        fontsize=7.5, color="#888888")
ax.text(loss_x, loss_y - 1.10, "  [PathMNIST only]", ha="left", va="center",
        fontsize=7, color="#aaaaaa", style="italic")

# ── DIMENSION annotation (right side, backbone) ───────────────────────────────
dim_x = CX + BW / 2 + 0.55
for yd, txt in [
    (y_input,  "(B, C, 28, 28)"),
    (y_stem,   "(B, 16, 28, 28)"),
    (y_uib1,   "(B, 24, 28, 28)"),
    (y_uib2,   "(B, 48, 14, 14)"),
    (y_uib3,   "(B, 96, 7, 7)"),
    (y_proj,   "(B, 49, 9)"),
    (y_ssa,    "(B, 49, 9)"),
    (y_measure,"(B, 9)"),
    (y_fusion, "(B, 64)"),
]:
    ax.text(dim_x, yd, txt, ha="left", va="center",
            fontsize=6.5, color="#999999", style="italic")

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_x, legend_y = 0.35, 5.10
legend_items = [
    (C_CLASSICAL, "Classical (UIB Backbone)"),
    (C_ATTN,      "Attention + Projection"),
    (C_QUANTUM,   "Quantum Circuit"),
    (C_FUSION,    "CQ Fusion"),
    (C_HEAD,      "Classification Head"),
]
ax.text(legend_x, legend_y + 0.42, "Legend", ha="left", va="center",
        fontsize=8, fontweight="bold", color="#333333")
for li, (lc, lt) in enumerate(legend_items):
    yl = legend_y - li * 0.38
    rect = mpatches.FancyBboxPatch(
        (legend_x - 0.05, yl - 0.13), 0.42, 0.28,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=lc, edgecolor="none", alpha=0.85, zorder=6,
    )
    ax.add_patch(rect)
    ax.text(legend_x + 0.50, yl, lt, ha="left", va="center",
            fontsize=7.5, color="#333333")

ax.text(legend_x + 0.50, legend_y - 5 * 0.38, "★  Novel contribution", ha="left",
        va="center", fontsize=7.5, color=C_NOVEL, fontweight="bold")

# ── Parameter count annotation ────────────────────────────────────────────────
pc_x, pc_y = CX + 2.8, 5.20
ax.text(pc_x, pc_y + 0.42, "Parameter Budget", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#333333")
params = [
    ("Classical backbone", "≈ 53 K"),
    ("UIB projector + SSA", "≈  9 K  ★"),
    ("Quantum VQC",        "≈  1 K"),
    ("CQ Fusion + Head",   "≈  8 K"),
    ("Total",              "≈ 71 K"),
]
for pi, (pn, pv) in enumerate(params):
    py = pc_y - pi * 0.35
    ax.text(pc_x - 1.3, py, pn, ha="left", va="center", fontsize=7, color="#555555")
    ax.text(pc_x + 1.3, py, pv, ha="right", va="center", fontsize=7,
            color=C_NOVEL if "★" in pv else "#333333", fontweight="bold")
ax.axhline(pc_y - 3.75 * 0.35 - 0.08, xmin=(pc_x - 1.45) / FIG_W,
           xmax=(pc_x + 1.45) / FIG_W, color="#cccccc", lw=0.8)

# ── Re-uploading annotation on quantum side ────────────────────────────────────
ru_x = CX + 0.3 + QW / 2 + 0.72
for lyr in range(1, 4):
    rl_y = y_qenc - (lyr - 1) * (y_qenc - y_hadamard) / 3
    ax.text(ru_x, rl_y, f"Layer {lyr}", ha="center", va="center",
            fontsize=6.5, color=C_QUANTUM, alpha=0.75)
ax.text(ru_x, (y_qenc + y_hadamard) / 2 + 0.45, "L=3\nre-upload",
        ha="center", va="center", fontsize=7, color=C_QUANTUM,
        fontweight="bold", alpha=0.9)
ax.annotate("", xy=(ru_x - 0.38, y_hadamard + 0.1),
            xytext=(ru_x - 0.38, y_qenc - 0.1),
            arrowprops=dict(arrowstyle="<->", color=C_QUANTUM,
                            lw=1.0, mutation_scale=8, alpha=0.6))

# ── Final polish ───────────────────────────────────────────────────────────────
ax.text(CX, 0.30, "AQHM-Net  ·  Attention-Guided Hybrid Quantum MobileNet  ·  28×28 input  ·  11 qubits  ·  L=3 layers",
        ha="center", va="center", fontsize=7, color="#aaaaaa")

plt.tight_layout(pad=0)
for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"aqhm_net_architecture.{ext}")
    plt.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")

plt.close()
