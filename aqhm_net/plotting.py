"""
plotting.py
-----------
Publication-quality visualisations for AQHM-Net experiments.

Output per dataset (results/<dataset>/plots/):
  training_curves.png    — loss + accuracy vs epoch (all seeds + mean)
  confusion_matrix.png   — row-normalised confusion matrix (best seed)
  per_class_auc.png      — per-class AUC-ROC bars (best seed)
  metrics_boxplot.png    — metric distributions across seeds
  loss_vs_seed.png       — best val-loss per seed (variance check)

Root output:
  all_datasets_comparison.png — cross-dataset weighted-F1 comparison
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


# ---------------------------------------------------------------------------
# Publication style — IEEE / Nature single-column compatible
# ---------------------------------------------------------------------------

_FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family":           _FONT,
    "font.size":             9,
    "axes.titlesize":        9,
    "axes.titleweight":      "normal",
    "axes.labelsize":        9,
    "axes.labelweight":      "normal",
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    "legend.framealpha":     0.85,
    "legend.edgecolor":      "#cccccc",
    "legend.borderpad":      0.4,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.7,
    "axes.grid":             True,
    "grid.color":            "#e8e8e8",
    "grid.linewidth":        0.5,
    "grid.alpha":            1.0,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "lines.linewidth":       1.4,
    "patch.linewidth":       0.5,
})

# Okabe–Ito colorblind-safe palette (8 hues) — print-safe
_PALETTE = [
    "#0072B2",   # blue
    "#E69F00",   # orange
    "#009E73",   # green
    "#CC79A7",   # pink/purple
    "#56B4E9",   # sky blue
    "#D55E00",   # red-orange
    "#F0C030",   # yellow (lightened)
    "#555555",   # dark grey
    "#88CCEE",   # light teal
    "#AA4499",   # purple
]

# Accent for reference lines / mean
_BLACK  = "#1a1a1a"
_GREY   = "#999999"
_LGREY  = "#e0e0e0"

# Dataset display names
_DS_LABELS = {
    "mnist_0123":    "MNIST-4",
    "mnist":         "MNIST-10",
    "pneumoniamnist":"PneumoniaMNIST",
    "breastmnist":   "BreastMNIST",
    "dermamnist":    "DermaMNIST",
    "pathmnist":     "PathMNIST",
}


def _col(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [plot] -> {path}")


def _ds_label(name: str) -> str:
    return _DS_LABELS.get(name, name)


# ---------------------------------------------------------------------------
# 1. Training curves — loss & accuracy (all seeds + shaded band + mean)
# ---------------------------------------------------------------------------

def plot_training_curves(
    histories: list[dict],
    output_dir: str,
    dataset_name: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for ax, (train_key, val_key), ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss",                      "Accuracy"],
    ):
        max_ep = max(len(h[val_key]) for h in histories)
        ep = np.arange(1, max_ep + 1)

        # Individual seed traces — thin, semi-transparent
        for i, h in enumerate(histories):
            n = len(h[val_key])
            ax.plot(
                range(1, n + 1), h[val_key],
                color=_col(i), lw=0.8, alpha=0.35,
            )

        # Shaded band (mean ± std across seeds)
        padded_val = np.array([
            h[val_key] + [h[val_key][-1]] * (max_ep - len(h[val_key]))
            for h in histories
        ])
        mean_v = padded_val.mean(axis=0)
        std_v  = padded_val.std(axis=0)

        ax.fill_between(ep, mean_v - std_v, mean_v + std_v,
                        color=_col(0), alpha=0.12)
        ax.plot(ep, mean_v, color=_col(0), lw=1.8, label="val mean")

        # Mean train as dashed
        padded_trn = np.array([
            h[train_key] + [h[train_key][-1]] * (max_ep - len(h[train_key]))
            for h in histories
        ])
        mean_t = padded_trn.mean(axis=0)
        ax.plot(ep, mean_t, color=_col(0), lw=1.0, ls="--",
                alpha=0.65, label="train mean")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    ds = _ds_label(dataset_name)
    axes[0].set_title(f"{ds}  —  loss")
    axes[1].set_title(f"{ds}  —  accuracy")

    fig.tight_layout(w_pad=1.5)
    _save(fig, os.path.join(output_dir, "training_curves.png"))


# ---------------------------------------------------------------------------
# 2. Confusion matrix — row-normalised (best seed)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: Optional[list[str]],
    output_dir: str,
    dataset_name: str = "",
    seed_idx: int = 0,
) -> None:
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_arr / row_sums

    n = cm_arr.shape[0]
    sz = max(3.2, n * 0.52 + 1.2)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.90))
    ax.grid(False)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=class_names or [str(i) for i in range(n)],
    )
    disp.plot(
        ax=ax, colorbar=True, cmap="Blues",
        values_format=".2f",
        im_kw={"vmin": 0.0, "vmax": 1.0},
    )

    ds = _ds_label(dataset_name)
    ax.set_title(f"{ds}  —  confusion matrix (seed {seed_idx})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7.5)
    plt.setp(ax.get_yticklabels(), fontsize=7.5)

    # Clean colourbar tick labels
    cbar = disp.im_.colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Recall", fontsize=7.5)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "confusion_matrix.png"))


# ---------------------------------------------------------------------------
# 3. Per-class AUC-ROC — horizontal bars (best seed)
# ---------------------------------------------------------------------------

def plot_per_class_auc(
    per_class_auc: list[float],
    class_names: Optional[list[str]],
    output_dir: str,
    dataset_name: str = "",
) -> None:
    n      = len(per_class_auc)
    labels = class_names or [str(i) for i in range(n)]
    aucs   = np.array(per_class_auc, dtype=float)
    order  = np.argsort(aucs)   # ascending so best is at top after invert

    fig, ax = plt.subplots(figsize=(5.2, max(2.4, n * 0.38 + 0.8)))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    colors = [_col(0) if a >= 0.90 else _col(1) if a >= 0.80 else _col(5)
              for a in aucs[order]]

    bars = ax.barh(
        [labels[i] for i in order], aucs[order],
        color=colors, edgecolor="white", height=0.60,
    )

    for bar, v in zip(bars, aucs[order]):
        x_pos = min(v + 0.006, 0.975)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=7.5)

    ax.axvline(0.90, color=_col(0), lw=0.9, ls="--", alpha=0.7)
    ax.axvline(0.80, color=_col(1), lw=0.9, ls="--", alpha=0.7)
    ax.set_xlim(max(0.3, aucs.min() - 0.08), 1.05)
    ax.set_xlabel("AUC-ROC (one-vs-rest)")

    legend_handles = [
        matplotlib.patches.Patch(color=_col(0), label=r"$\geq$0.90"),
        matplotlib.patches.Patch(color=_col(1), label=r"$\geq$0.80"),
        matplotlib.patches.Patch(color=_col(5), label="<0.80"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5)

    ds = _ds_label(dataset_name)
    ax.set_title(f"{ds}  —  per-class AUC-ROC")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "per_class_auc.png"))


# ---------------------------------------------------------------------------
# 4. Metrics boxplot — all seeds
# ---------------------------------------------------------------------------

def plot_metrics_boxplot(
    per_run_metrics: list[dict],
    output_dir: str,
    dataset_name: str = "",
) -> None:
    metric_keys   = ["weighted_f1", "macro_f1", "weighted_accuracy", "mean_auc"]
    metric_labels = ["Weighted\nF1", "Macro\nF1", "Weighted\nAccuracy", "Mean\nAUC"]

    data = [
        [m[k] for m in per_run_metrics if not np.isnan(m.get(k, float("nan")))]
        for k in metric_keys
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.set_axisbelow(True)
    positions = list(range(1, len(metric_keys) + 1))

    bp = ax.boxplot(
        data, positions=positions, widths=0.42,
        patch_artist=True, notch=False,
        medianprops=dict(color=_BLACK, lw=1.8),
        whiskerprops=dict(lw=0.9, color="#555555"),
        capprops=dict(lw=0.9, color="#555555"),
        flierprops=dict(marker="x", markersize=4, color=_col(5), lw=0.8),
    )
    box_colors = [_col(0), _col(2), _col(1), _col(3)]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.40)

    # Jitter individual seed points
    rng = np.random.default_rng(42)
    for pos, vals, c in zip(positions, data, box_colors):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            [pos + j for j in jitter], vals,
            color=c, edgecolors=_BLACK,
            s=22, zorder=5, lw=0.6, alpha=0.85,
        )

    y_min = max(0.0, min(v for d in data for v in d) - 0.04)
    ax.set_ylim(y_min, 1.03)
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.axhline(1.0, color=_GREY, lw=0.5, ls="--")

    ds = _ds_label(dataset_name)
    n_s = len(per_run_metrics)
    ax.set_title(f"{ds}  —  {n_s} seeds")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "metrics_boxplot.png"))


# ---------------------------------------------------------------------------
# 5. Best val-loss per seed — scatter + mean band
# ---------------------------------------------------------------------------

def plot_loss_per_seed(
    histories: list[dict],
    output_dir: str,
    dataset_name: str = "",
) -> None:
    best_losses = [h.get("best_val_loss", h["val_loss"][-1]) for h in histories]
    seeds = list(range(len(best_losses)))

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    mu  = np.mean(best_losses)
    sig = np.std(best_losses)
    ax.fill_between(seeds, mu - sig, mu + sig,
                    color=_col(0), alpha=0.12, label=r"$\pm$1 s.d.")
    ax.axhline(mu, color=_col(0), lw=1.3, ls="-",
               label=f"mean = {mu:.4f}")
    ax.plot(seeds, best_losses, color=_GREY, lw=0.8, ls="--", zorder=3)
    ax.scatter(seeds, best_losses,
               color=[_col(i) for i in seeds],
               edgecolors="white", s=55, zorder=5, lw=0.7)

    ax.set_xlabel("Seed index")
    ax.set_ylabel("Best val loss")
    ax.set_xticks(seeds)
    ax.legend(loc="best")

    ds = _ds_label(dataset_name)
    ax.set_title(f"{ds}  —  best val loss per seed")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "loss_vs_seed.png"))


# ---------------------------------------------------------------------------
# 6. Cross-dataset comparison — grouped bars (W-F1 + W-Acc)
# ---------------------------------------------------------------------------

def plot_dataset_comparison(
    summaries: dict[str, dict],
    output_dir: str,
) -> None:
    ds_order = ["mnist_0123", "mnist", "pneumoniamnist",
                "breastmnist", "dermamnist", "pathmnist"]
    names = [n for n in ds_order if n in summaries] + \
            [n for n in summaries if n not in ds_order]

    f1s    = [summaries[n].get("weighted_f1_mean",  summaries[n].get("val_acc_mean", 0)) for n in names]
    f1_std = [summaries[n].get("weighted_f1_std",   summaries[n].get("val_acc_std",  0)) for n in names]
    accs   = [summaries[n].get("val_acc_mean",  0) for n in names]
    acc_std= [summaries[n].get("val_acc_std",   0) for n in names]

    x      = np.arange(len(names))
    w      = 0.36
    labels = [_ds_label(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(5.0, len(names) * 1.25), 3.2))
    ax.set_axisbelow(True)

    bars_f1 = ax.bar(
        x - w / 2, f1s, w,
        yerr=f1_std, color=_col(0), alpha=0.80, edgecolor="white",
        capsize=3.5, error_kw=dict(elinewidth=0.9, ecolor="#333333"),
        label="Weighted F1",
    )
    bars_ac = ax.bar(
        x + w / 2, accs, w,
        yerr=acc_std, color=_col(1), alpha=0.80, edgecolor="white",
        capsize=3.5, error_kw=dict(elinewidth=0.9, ecolor="#333333"),
        label="Weighted Accuracy",
    )

    # Value annotations
    for bar, v, e in zip(bars_f1, f1s, f1_std):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + e + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
    for bar, v, e in zip(bars_ac, accs, acc_std):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + e + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.10)
    ax.axhline(1.0, color=_GREY, lw=0.5, ls="--")
    ax.legend(loc="lower right")
    ax.set_title("AQHM-Net  —  results across datasets (mean ± s.d.)")

    fig.tight_layout()
    path = os.path.join(output_dir, "all_datasets_comparison.png")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Convenience: generate all plots for one completed dataset run
# ---------------------------------------------------------------------------

def save_all_plots(
    histories: list[dict],
    per_run_metrics: list[dict],
    output_dir: str,
    dataset_name: str,
    class_names: Optional[list[str]] = None,
    best_seed_idx: int = 0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[plots] Generating -> {output_dir}")

    plot_training_curves(histories, output_dir, dataset_name)

    best_cm = per_run_metrics[best_seed_idx].get("confusion_matrix")
    if best_cm is not None:
        plot_confusion_matrix(best_cm, class_names, output_dir,
                              dataset_name, best_seed_idx)

    best_auc = per_run_metrics[best_seed_idx].get("per_class_auc", [])
    if best_auc and not np.isnan(best_auc[0]):
        plot_per_class_auc(best_auc, class_names, output_dir, dataset_name)

    plot_metrics_boxplot(per_run_metrics, output_dir, dataset_name)
    plot_loss_per_seed(histories, output_dir, dataset_name)

    print(f"[plots] Done  -> {output_dir}")
