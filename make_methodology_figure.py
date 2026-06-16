"""
make_methodology_figure.py
---------------------------
Renders the AQHM-Net architecture as a publication-quality methodology figure.

Pipeline (serpentine, left -> right then wrapping down):
  Row 1  Input -> Stem -> UIB+SE -> UIB+CBAM -> UIB+SE -> feature map (7x7x96)
  Row 2  Superpixel projector -> Spatial Superpixel Attention (SSA)
         -> Quantum encoding (ql/qe) -> Quantum conv (4 heterogeneous kernels)
  Row 3  Hadamard integration + re-uploading -> Measurement (6 PauliX + 3 ZZ)
         -> Soft alpha-gate fusion (with classical context z_c) -> Classifier

Output: results/methodology_architecture.png (+ .pdf), 300 DPI.
Style matches aqhm_net/plotting.py: Okabe-Ito palette, DejaVu Sans, no big titles.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style (mirrors aqhm_net/plotting.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8.5,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.10,
})

# Okabe-Ito colorblind-safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
PINK   = "#CC79A7"
PURPLE = "#AA4499"
GREY   = "#555555"
INK    = "#1a1a1a"
LINEC  = "#444444"

# Soft fills for the four stage bands (light tints of the group hues)
BAND_CLASSICAL = "#eaf3fb"
BAND_BRIDGE    = "#e8f6f1"
BAND_QUANTUM   = "#fdf3e3"
BAND_HEAD      = "#f8edf4"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, title, sub=None, edge=INK, face="white",
        fs=8.5, sub_fs=7.0, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.5",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=3))
    cx = x + w / 2.0
    if sub:
        ax.text(cx, y + h * 0.64, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=4,
                linespacing=1.05)
        ax.text(cx, y + h * 0.27, sub, ha="center", va="center",
                fontsize=sub_fs, color=GREY, zorder=4, linespacing=1.25)
    else:
        ax.text(cx, y + h / 2.0, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=4,
                linespacing=1.05)


def band(ax, x, y, w, h, color, label, label_color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none",
                            zorder=1))
    ax.text(x + 0.6, y + h - 0.55, label, ha="left", va="top",
            fontsize=8.0, fontweight="bold", color=label_color, zorder=2)


def arrow(ax, x0, y0, x1, y1, color=LINEC, lw=1.4, head=True, mut=11):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>" if head else "-", mutation_scale=mut,
        linewidth=lw, color=color, zorder=5, joinstyle="round"))


def tag(ax, x, y, text, color=GREY):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
            color=color, family="monospace", style="italic", zorder=6)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.8, 8.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 64)
ax.axis("off")

BH = 6.4
R1, R2, R3 = 47.0, 27.0, 6.0            # row baselines
mid = lambda y: y + BH / 2.0

# ====================  ROW 1 : CLASSICAL BACKBONE  =========================
band(ax, 1.5, R1 - 2.6, 97.0, BH + 4.6, BAND_CLASSICAL,
     "CLASSICAL BACKBONE   ·   MobileNetV4-inspired UIB + hierarchical attention", BLUE)

box(ax, 2.5, R1, 9.5, BH, "Input image",
    "MNIST·MedMNIST 28²\nCIFAR-10/-100 32²\n(native res.)", edge=GREY, sub_fs=6.0)
box(ax, 15.5, R1, 11.5, BH, "Stem", "Conv 3×3 + BN\n+ ReLU6", edge=BLUE)
box(ax, 30.5, R1, 14.5, BH, "Stage 1", "UIB 16→24 (t=4)\n+ SE  ·  28²/32² ×24",
    edge=BLUE, sub_fs=6.6)
box(ax, 48.5, R1, 15.5, BH, "Stage 2", "UIB 24→48 (t=6, s2)\n+ CBAM  ·  14²/16² ×48",
    edge=BLUE, sub_fs=6.6)
box(ax, 67.5, R1, 15.5, BH, "Stage 3", "UIB 48→96 (t=6, s2)\n+ SE  ·  7²/8² ×96",
    edge=BLUE, sub_fs=6.6)
box(ax, 86.5, R1, 11.0, BH, "Feature map", "7×7 / 8×8 ×96\n49 / 64 superpixels",
    edge=GREY, face="#f4f4f4", sub_fs=6.4)

for x0, x1 in [(12.0, 15.5), (27.0, 30.5), (45.0, 48.5), (64.0, 67.5), (83.0, 86.5)]:
    arrow(ax, x0, mid(R1), x1, mid(R1))

# Wrap: feature map -> projector (row 2)
arrow(ax, 92.0, R1 - 0.1, 92.0, R2 + BH + 0.3)
tag(ax, 95.6, (R1 + R2) / 2 + 4, "reshape\n(B,49,96)")

# ====================  ROW 2 : BRIDGE + QUANTUM CORE  ======================
band(ax, 1.5, R2 - 2.6, 65.0, BH + 4.6, BAND_QUANTUM,
     "QUANTUM CORE   ·   11 qubits  (6 qℓ + 3 qe + 2 ancilla)", ORANGE)
band(ax, 68.0, R2 - 2.6, 30.5, BH + 4.6, BAND_BRIDGE,
     "SUPERPIXEL BRIDGE   (novel)", GREEN)

box(ax, 87.0, R2, 11.0, BH, "UIB projector", "Lin 96→48→9\n(B, 49, 9)", edge=GREEN)
box(ax, 70.0, R2, 15.0, BH, "Spatial Superpixel\nAttention (SSA)",
    "SE over 49 / 64 patches\n→ weight each patch", edge=GREEN)
box(ax, 50.0, R2, 16.5, BH, "Quantum encoding",
    "qℓ: H + ctrl-U3 (49/64 pos)\nqe: U3 + CZ all-to-all", edge=ORANGE, sub_fs=6.6)
box(ax, 33.0, R2, 14.5, BH, "Quantum conv",
    "4 kernels via 2 ancilla\nA·U2 B·U3CB C·U4NN D·U4AA",
    edge=ORANGE, sub_fs=6.3)

arrow(ax, 87.0, mid(R2), 85.0, mid(R2))                       # projector -> SSA
arrow(ax, 70.0, mid(R2), 66.5, mid(R2))                       # SSA -> encoding
tag(ax, 68.2, mid(R2) - 4.4, "α·patch")
arrow(ax, 50.0, mid(R2), 47.5, mid(R2))                       # encoding -> qconv

# Native-resolution callout (fills empty left region of the quantum band)
ax.add_patch(FancyBboxPatch(
    (3.0, R2 + 0.1), 26.0, BH - 0.2,
    boxstyle="round,pad=0.012,rounding_size=0.5",
    linewidth=1.0, edgecolor=ORANGE, facecolor="white",
    linestyle=(0, (4, 2)), zorder=3))
ax.text(16.0, R2 + BH * 0.78, "Logarithmic location encoding",
        ha="center", va="center", fontsize=7.0, fontweight="bold",
        color=ORANGE, zorder=4)
ax.text(16.0, R2 + BH * 0.36,
        "28² → 7×7 = 49   ·   32² → 8×8 = 64 patches\n"
        "both fit 6 qℓ qubits (2⁶ = 64) → core stays\n"
        "at 11 qubits; native CIFAR adds no overhead",
        ha="center", va="center", fontsize=6.2, color=GREY,
        linespacing=1.3, zorder=4)

# Wrap: quantum conv -> Hadamard integration (row 3)
arrow(ax, 40.0, R2 - 0.1, 40.0, R3 + BH + 0.3)
tag(ax, 36.4, (R2 + R3) / 2 + 1.5, "superposed\nchannels")

# ====================  ROW 3 : INTEGRATION -> HEAD  ========================
band(ax, 1.5, R3 - 2.6, 53.5, BH + 4.6, BAND_QUANTUM,
     "INTEGRATION   ·   MEASUREMENT", ORANGE)
band(ax, 57.0, R3 - 2.6, 41.5, BH + 4.6, BAND_HEAD,
     "FUSION   ·   CLASSIFIER", PINK)

box(ax, 28.0, R3, 16.0, BH, "Hadamard integ.\n+ re-uploading",
    "post-select |00⟩\nL=3, per-layer proj.", edge=ORANGE)
box(ax, 8.0, R3, 17.0, BH, "Measurement",
    "6 × ⟨PauliX⟩ + 3 × ⟨ZZ⟩\n→ 9-dim quantum vec", edge=ORANGE)
box(ax, 59.0, R3, 18.0, BH, "Soft α-gate fusion",
    "h = α·h_q + (1−α)·h_c\nproj. 9→64 / 96→64", edge=PINK)
box(ax, 80.0, R3, 16.5, BH, "Classifier",
    "Lin 64→32 + ReLU + Drop(0.4)\n→ softmax  ·  C ∈ {2 … 100}", edge=PINK,
    sub_fs=6.4)

arrow(ax, 28.0, mid(R3), 25.0, mid(R3))                       # Hadamard -> measurement
arrow(ax, 77.0, mid(R3), 80.0, mid(R3))                       # fusion -> classifier
arrow(ax, 96.5, mid(R3), 98.6, mid(R3))                       # classifier -> output
ax.text(98.9, mid(R3), "class\nprobs.", ha="left", va="center",
        fontsize=7.2, fontweight="bold", color=INK)

# z_q : measurement -> fusion (along the bottom corridor)
for seg in [(16.5, R3 - 0.1, 16.5, R3 - 2.5), (16.5, R3 - 2.5, 63.0, R3 - 2.5),
            (63.0, R3 - 2.5, 63.0, R3 - 0.1)]:
    arrow(ax, *seg, head=False, lw=1.3)
arrow(ax, 63.0, R3 - 0.6, 63.0, R3 - 0.1, lw=1.3)             # arrowhead into fusion
tag(ax, 39.5, R3 - 3.05, "z_q   (9-dim quantum features)")

# z_c : backbone (feature map) -> fusion top, via far-right corridor (blue)
zc = BLUE
for seg in [(97.5, R1, 99.2, R1), (99.2, R1, 99.2, 19.0),
            (99.2, 19.0, 68.0, 19.0), (68.0, 19.0, 68.0, R3 + BH + 0.3)]:
    arrow(ax, *seg, head=False, lw=1.2, color=zc)
arrow(ax, 68.0, R3 + BH + 0.8, 68.0, R3 + BH + 0.1, lw=1.2, color=zc)
ax.text(83.0, 19.9, "z_c   (96-dim classical context, global avg-pool)",
        ha="center", va="bottom", fontsize=6.5, color=zc,
        family="monospace", style="italic")

# ---------------------------------------------------------------------------
# Title + legend
# ---------------------------------------------------------------------------
ax.text(1.5, 63.2, "AQHM-Net", fontsize=13.5, fontweight="bold", color=INK,
        ha="left", va="top")
ax.text(1.5, 60.0,
        "Attention-Guided Hybrid Quantum MobileNet — end-to-end architecture",
        fontsize=8.8, color=GREY, ha="left", va="top")

legend_items = [("Classical backbone", BLUE), ("Superpixel bridge (novel)", GREEN),
                ("Quantum core", ORANGE), ("Fusion / classifier", PINK)]
handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                  markerfacecolor=c, markeredgecolor=c, label=lbl)
           for lbl, c in legend_items]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.02),
          ncol=4, frameon=False, fontsize=7.8, handletextpad=0.4,
          columnspacing=1.3)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
png = os.path.join(out_dir, "methodology_architecture.png")
pdf = os.path.join(out_dir, "methodology_architecture.pdf")
fig.savefig(png, dpi=300)
fig.savefig(pdf)
print(f"Saved:\n  {png}\n  {pdf}")
