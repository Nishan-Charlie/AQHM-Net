"""Compile every summary.json on disk into a single results_summary.csv.

One row per experiment (one summary.json). All `results*` directories are
auto-discovered, so newly added experiment groups are picked up automatically.
Metadata (dataset, K heads, attention-encoding, scale, scheduler, augmentation,
resolution) is parsed from the experiment folder name.
"""
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

KNOWN_DATASETS = [
    "pneumoniamnist", "breastmnist", "dermamnist", "pathmnist",
    "cifar100", "cifar10", "mnist_0123", "mnist",
]
KNOWN_SCALES = ["micro", "small", "medium", "large"]
KNOWN_SCHEDULERS = ["onecycle", "cosine"]
METRICS = ["weighted_accuracy", "weighted_f1", "macro_f1", "mean_auc"]


def fmt(x):
    """Render a value, mapping NaN/None to empty so the CSV stays clean."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x


def parse_meta(folder: str):
    """Pull dataset + experiment knobs out of an experiment folder name."""
    dataset = next((d for d in KNOWN_DATASETS if folder.startswith(d)), folder)
    k_match = re.search(r"_k(\d+)", folder)
    cutmix = re.search(r"cutmix([\d.]+)", folder)
    mixup = re.search(r"mixup([\d.]+)", folder)
    res = re.search(r"_r(\d+)", folder)
    return {
        "dataset": dataset,
        "quantum_heads_K": int(k_match.group(1)) if k_match else "",
        "attention_encoding": "_attenc" in folder,
        "no_quantum": "no_quantum" in folder,
        "scale": next((s for s in KNOWN_SCALES if f"_{s}" in folder), ""),
        "scheduler": next((s for s in KNOWN_SCHEDULERS if s in folder), ""),
        "cutmix_alpha": cutmix.group(1) if cutmix else "",
        "mixup_alpha": mixup.group(1) if mixup else "",
        "resolution": res.group(1) if res else "",
    }


def discover_dirs():
    """Every results* subdirectory of ROOT, sorted, dirs only."""
    return sorted(
        p for p in ROOT.glob("results*") if p.is_dir()
    )


rows = []
for base in discover_dirs():
    for summary_path in sorted(base.glob("*/summary.json")):
        folder = summary_path.parent.name
        try:
            data = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        meta = parse_meta(folder)
        row = {
            "experiment_group": base.name,
            "dataset": meta["dataset"],
            "folder_name": folder,
            "quantum_heads_K": meta["quantum_heads_K"],
            "attention_encoding": meta["attention_encoding"],
            "no_quantum": meta["no_quantum"],
            "scale": meta["scale"],
            "scheduler": meta["scheduler"],
            "cutmix_alpha": meta["cutmix_alpha"],
            "mixup_alpha": meta["mixup_alpha"],
            "resolution": meta["resolution"],
            "n_runs": data.get("n_runs", ""),
            "best_seed_idx": data.get("best_seed_idx", ""),
        }
        for m in METRICS:
            block = data.get(m, {}) or {}
            row[f"{m}_mean"] = fmt(block.get("mean"))
            row[f"{m}_std"] = fmt(block.get("std"))
            row[f"{m}_min"] = fmt(block.get("min"))
            row[f"{m}_max"] = fmt(block.get("max"))
        row["source_path"] = summary_path.relative_to(ROOT).as_posix()
        rows.append(row)

# Sort for readability: dataset, then group, then folder.
rows.sort(key=lambda r: (r["dataset"], r["experiment_group"], r["folder_name"]))

fieldnames = [
    "experiment_group", "dataset", "folder_name",
    "quantum_heads_K", "attention_encoding", "no_quantum",
    "scale", "scheduler", "cutmix_alpha", "mixup_alpha", "resolution",
    "n_runs", "best_seed_idx",
]
for m in METRICS:
    fieldnames += [f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max"]
fieldnames.append("source_path")

out = ROOT / "results_summary.csv"
with out.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} experiment rows to {out.name}")
groups = sorted({r["experiment_group"] for r in rows})
print(f"Groups ({len(groups)}): {', '.join(groups)}")
