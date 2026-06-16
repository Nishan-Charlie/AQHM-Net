"""
evaluate.py
-----------
Evaluation metrics and statistical analysis for AQHM-Net experiments.

Section 15 of METHODOLOGY.md:

15.1 — Metrics:
    Primary:   Weighted F1-Score (appropriate for MedMNIST class imbalance)
    Secondary: Weighted Accuracy, Precision, Recall
               Per-class AUC-ROC (one-vs-rest, medical datasets)
               Confusion matrix (full N×N)

    NOTE: Overall accuracy is insufficient for imbalanced datasets
    (e.g., a majority-class predictor can reach 67% on ISIC 2018).
    Macro-F1 and per-class recall are co-primary metrics.

15.2 — Statistical reporting:
    All results: mean ± std across 10 independent runs.
    Significance: Wilcoxon signed-rank test (non-parametric, n=10).
    Multiple comparisons: Bonferroni correction.
    Effect size: Cohen's d.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from scipy import stats

from .model import AQHMNet


# ---------------------------------------------------------------------------
# Core per-run evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test-time augmentation — exact (interpolation-free) symmetry views
# ---------------------------------------------------------------------------

def _tta_view(x: torch.Tensor, view: str) -> torch.Tensor:
    """Apply one exact symmetry transform to a (B, C, H, W) batch.

    All views are pixel-exact (flips / 90° rotations) — no interpolation and no
    border fill, so they introduce zero artefacts on the normalised tensor.
    Only valid for orientation-agnostic imagery (medical), never for digits.
    """
    if view == "id":     return x
    if view == "hflip":  return torch.flip(x, dims=[-1])
    if view == "vflip":  return torch.flip(x, dims=[-2])
    if view == "rot90":  return torch.rot90(x, k=1, dims=[-2, -1])
    if view == "rot180": return torch.rot90(x, k=2, dims=[-2, -1])
    if view == "rot270": return torch.rot90(x, k=3, dims=[-2, -1])
    raise ValueError(f"Unknown TTA view '{view}'")


@torch.no_grad()
def evaluate_model(
    model: AQHMNet,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    num_classes: int = 10,
    tta_views: Optional[list[str]] = None,
) -> dict:
    """Compute all evaluation metrics for one model checkpoint.

    Args:
        model       : trained AQHMNet.
        loader      : test DataLoader.
        device      : torch device; auto-detected if None.
        num_classes : number of output classes.
        tta_views   : if given, list of exact symmetry views (e.g.
                      ["id", "hflip", "vflip", "rot180"]) whose softmax outputs
                      are averaged per sample — test-time augmentation. None or
                      ["id"] disables TTA. Use only for orientation-agnostic
                      (medical) data, never for MNIST digits.

    Returns:
        metrics dict with:
            "weighted_f1"        : weighted F1-score (primary)
            "macro_f1"           : macro-averaged F1
            "weighted_accuracy"  : weighted accuracy
            "weighted_precision" : weighted precision
            "weighted_recall"    : weighted recall
            "per_class_auc"      : list of per-class AUC-ROC (OvR)
            "mean_auc"           : mean AUC-ROC
            "confusion_matrix"   : N×N array
            "report"             : sklearn classification_report string
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_preds   = []
    all_labels  = []
    all_probs   = []   # for AUC-ROC

    views = tta_views if tta_views else ["id"]

    def _forward(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        return out[0] if model.use_contrastive else out

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.cpu().squeeze().long()

        # Average softmax over exact symmetry views (TTA). Single "id" view
        # reduces to ordinary evaluation.
        probs = torch.zeros(imgs.size(0), num_classes, device=device)
        for v in views:
            probs += F.softmax(_forward(_tta_view(imgs, v)), dim=-1)
        probs = (probs / len(views)).cpu()
        preds = probs.argmax(dim=-1)

        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(probs)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.cat(all_probs).numpy()   # (N, num_classes)

    # ── Weighted metrics (handle class imbalance) ──────────────────────────
    w_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    m_f1   = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    w_prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    w_rec  = recall_score(all_labels, all_preds, average="weighted",    zero_division=0)
    w_acc  = accuracy_score(all_labels, all_preds)

    # ── Per-class AUC-ROC (one-vs-rest) ────────────────────────────────────
    # Only meaningful when at least 2 classes present in the test set
    try:
        if num_classes == 2:
            auc_per_class = [
                roc_auc_score(all_labels, all_probs[:, 1])
            ]
        else:
            auc_per_class = roc_auc_score(
                all_labels, all_probs,
                multi_class="ovr", average=None
            ).tolist()
        mean_auc = float(np.mean(auc_per_class))
    except ValueError:
        # Edge case: only 1 class present in a very small test set
        auc_per_class = [float("nan")]
        mean_auc = float("nan")

    # ── Confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds).tolist()

    # ── Classification report (text) ───────────────────────────────────────
    report = classification_report(all_labels, all_preds, zero_division=0)

    return {
        "weighted_f1":        float(w_f1),
        "macro_f1":           float(m_f1),
        "weighted_accuracy":  float(w_acc),
        "weighted_precision": float(w_prec),
        "weighted_recall":    float(w_rec),
        "per_class_auc":      auc_per_class,
        "mean_auc":           mean_auc,
        "confusion_matrix":   cm,
        "report":             report,
    }


# ---------------------------------------------------------------------------
# Statistical significance tests (Section 15.2)
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict:
    """Wilcoxon signed-rank test comparing two lists of per-run scores.

    Non-parametric test appropriate for small samples (n=10).
    Applied to compare AQHM-Net vs each baseline (Section 15.2).

    Args:
        scores_a : metric values for model A (e.g., AQHM-Net F1s).
        scores_b : metric values for model B (e.g., baseline F1s).
        alpha    : significance level (default 0.05).

    Returns:
        dict with "statistic", "p_value", "significant", "effect_d" (Cohen's d).
    """
    assert len(scores_a) == len(scores_b), "Both lists must have equal length"

    stat, p_val = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")

    # Cohen's d effect size
    diff = np.array(scores_a) - np.array(scores_b)
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)

    return {
        "statistic": float(stat),
        "p_value":   float(p_val),
        "significant": bool(p_val < alpha),
        "effect_d":  float(d),
    }


def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Apply Bonferroni correction to a list of p-values.

    Section 15.2: correction applied across all pairwise baseline comparisons.

    Args:
        p_values : list of raw p-values from wilcoxon_test calls.
        alpha    : family-wise error rate.

    Returns:
        List of booleans — True if the comparison is significant after correction.
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute Cohen's d effect size between two paired score lists."""
    diff = np.array(scores_a) - np.array(scores_b)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


# ---------------------------------------------------------------------------
# Multi-run result aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(
    per_run_metrics: list[dict],
    metric_key: str = "weighted_f1",
) -> dict:
    """Aggregate metrics from N independent runs (Section 15.2).

    Args:
        per_run_metrics : list of metric dicts from evaluate_model().
        metric_key      : which metric to aggregate.

    Returns:
        dict with "mean", "std", "min", "max", "values".
    """
    values = [m[metric_key] for m in per_run_metrics]
    return {
        "mean":   float(np.mean(values)),
        "std":    float(np.std(values, ddof=1)),
        "min":    float(np.min(values)),
        "max":    float(np.max(values)),
        "values": values,
    }


# ---------------------------------------------------------------------------
# Full ablation evaluation table
# ---------------------------------------------------------------------------

def evaluate_ablation_suite(
    ablation_results: dict[str, list[dict]],
    full_model_key: str = "AQHM-Net",
    output_path: Optional[str] = None,
) -> dict:
    """Evaluate and compare all ablations against the full AQHM-Net model.

    Args:
        ablation_results: {model_name: [metrics_run_0, ..., metrics_run_9]}
        full_model_key  : key for the reference model in ablation_results.
        output_path     : if provided, save the table as JSON.

    Returns:
        Summary dict with per-model stats and significance tests.
    """
    full_scores = [m["weighted_f1"] for m in ablation_results[full_model_key]]

    table = {}
    p_values = []

    for name, run_metrics in ablation_results.items():
        agg = aggregate_runs(run_metrics, "weighted_f1")
        table[name] = {"weighted_f1": agg}

        if name != full_model_key:
            abl_scores = [m["weighted_f1"] for m in run_metrics]
            wt = wilcoxon_test(full_scores, abl_scores)
            table[name]["vs_full"] = wt
            p_values.append(wt["p_value"])
        else:
            table[name]["vs_full"] = None

    # Bonferroni correction over all pairwise comparisons
    # (exclude the full model itself from correction)
    sig_corrected = bonferroni_correct(p_values)
    non_full = [k for k in table if k != full_model_key]
    for name, sig in zip(non_full, sig_corrected):
        if table[name]["vs_full"] is not None:
            table[name]["vs_full"]["significant_bonferroni"] = sig

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(table, f, indent=2)

    return table


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_results_table(
    per_run_metrics: list[dict],
    dataset_name: str = "dataset",
    model_name:   str = "AQHM-Net",
) -> None:
    """Print a formatted results table for one set of runs."""
    keys = ["weighted_f1", "macro_f1", "weighted_accuracy", "mean_auc"]
    labels = ["W-F1", "M-F1", "W-Acc", "mAUC"]

    print(f"\n{'-'*60}")
    print(f"{model_name}  |  {dataset_name}  |  {len(per_run_metrics)} runs")
    print(f"{'-'*60}")

    for key, label in zip(keys, labels):
        agg = aggregate_runs(per_run_metrics, key)
        print(
            f"  {label:<8}: {agg['mean']:.4f} ± {agg['std']:.4f}  "
            f"[{agg['min']:.4f} – {agg['max']:.4f}]"
        )
    print(f"{'-'*60}\n")
