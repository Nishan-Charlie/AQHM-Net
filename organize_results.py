"""Organise the sprawling results into one clean, deduplicated view.

The raw `results_summary.csv` (produced by compile_results_summary.py) contains
one row per summary.json across ~14 `results*` experiment groups. The same
nominal model configuration is frequently re-run in several groups (e.g. the
CIFAR-10 base config appears in `results`, `results_kcompare`, `results_parallel`
and `results_scale`), and a few runs are degenerate (mode-collapse: weighted-F1
near zero from label-blending augmentation on DermaMNIST, or the failed 10-class
MNIST base run).

This script collapses those duplicates to the single best instance per
configuration, flags degenerate runs, groups everything by domain -> dataset,
and writes three artefacts to `results_organized/`:

    all_configs.csv        deduplicated, one row per (dataset, config), sorted
    best_per_dataset.csv   the single best (non-degenerate) config per dataset
    README.md              human-readable report + provenance

Nothing on disk is moved or deleted; only derived summaries are written.
Re-run after adding experiments:  python organize_results.py
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results_summary.csv"
OUT_DIR = ROOT / "results_organized"

METRICS = ["weighted_accuracy", "weighted_f1", "macro_f1", "mean_auc"]

# Domain grouping for a tidy, reader-friendly ordering.
DOMAIN = {
    "mnist": "Digits", "mnist_0123": "Digits",
    "pathmnist": "Medical", "dermamnist": "Medical",
    "pneumoniamnist": "Medical", "breastmnist": "Medical",
    "cifar10": "Natural", "cifar100": "Natural",
}
DOMAIN_ORDER = {"Digits": 0, "Medical": 1, "Natural": 2}
DATASET_PRETTY = {
    "mnist": "MNIST (10-class)", "mnist_0123": "MNIST-4 (0-3)",
    "pathmnist": "PathMNIST", "dermamnist": "DermaMNIST",
    "pneumoniamnist": "PneumoniaMNIST", "breastmnist": "BreastMNIST",
    "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
}

# A run is treated as degenerate (training collapse) below this weighted-F1.
# The lowest legitimate result on file is CIFAR-100 (~0.36); collapses sit at
# <0.14, so 0.20 cleanly separates them.
DEGENERATE_WF1 = 0.20


def to_float(x: str):
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except ValueError:
        return None


def epochs_of(r: dict) -> int:
    """Training budget is not in the CSV; the only non-default budget is the
    dedicated `results_300ep` group, so derive it from the experiment group."""
    return 300 if r["experiment_group"] == "results_300ep" else 50


def canonical_config(r: dict) -> tuple:
    """Normalise the knobs that define a configuration (seed/group aside).

    Empty cells map to their defaults: K=1, scale=small, scheduler=cosine,
    cutmix/mixup=0, resolution=28. The training budget (50 vs 300 epochs) is
    part of the identity so that rigorous multi-seed 50-epoch runs are kept as
    their own row rather than being hidden behind a single 300-epoch run.
    """
    k = r["quantum_heads_K"] or "1"
    return (
        r["dataset"],
        int(float(k)),
        r["attention_encoding"].strip().lower() == "true",
        r["no_quantum"].strip().lower() == "true",
        r["scale"] or "small",
        r["scheduler"] or "cosine",
        r["cutmix_alpha"] or "0",
        r["mixup_alpha"] or "0",
        r["resolution"] or "28",
        epochs_of(r),
    )


def config_label(cfg: tuple) -> str:
    """Compact human-readable config string, e.g. 'K4+AE, large, r224, onecycle'."""
    (_ds, k, ae, noq, scale, sched, cutmix, mixup, res, epochs) = cfg
    parts = [f"K{k}" + ("+AE" if ae else "")]
    if noq:
        parts[0] = "no-quantum"
    if scale != "small":
        parts.append(scale)
    if res != "28":
        parts.append(f"r{res}")
    if sched != "cosine":
        parts.append(sched)
    aug = []
    if cutmix != "0":
        aug.append(f"cutmix{cutmix}")
    if mixup != "0":
        aug.append(f"mixup{mixup}")
    if aug:
        parts.append("+".join(aug))
    if epochs != 50:
        parts.append(f"{epochs}ep")
    return ", ".join(parts)


def better(a: dict, b: dict) -> dict:
    """Pick the better of two rows for the SAME config: higher weighted-F1,
    tie-broken by more seeds then higher weighted accuracy. Non-degenerate
    always beats degenerate."""
    if a["_degenerate"] != b["_degenerate"]:
        return a if not a["_degenerate"] else b
    ka = (a["_wf1"] or -1, a["_n"], a["_wacc"] or -1)
    kb = (b["_wf1"] or -1, b["_n"], b["_wacc"] or -1)
    return a if ka >= kb else b


def main() -> None:
    rows = list(csv.DictReader(SRC.open(newline="")))
    for r in rows:
        r["_wf1"] = to_float(r["weighted_f1_mean"])
        r["_wacc"] = to_float(r["weighted_accuracy_mean"])
        r["_n"] = int(r["n_runs"]) if r["n_runs"] else 1
        r["_degenerate"] = (r["_wf1"] is not None and r["_wf1"] < DEGENERATE_WF1)

    # ── Deduplicate by canonical config, keeping the best instance ───────────
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[canonical_config(r)].append(r)

    organised = []
    for cfg, members in groups.items():
        best = members[0]
        for m in members[1:]:
            best = better(best, m)
        collapsed = [m for m in members if m is not best]
        organised.append({
            "domain": DOMAIN.get(cfg[0], "Other"),
            "dataset": cfg[0],
            "config": config_label(cfg),
            "K": cfg[1],
            "attention_encoding": cfg[2],
            "scale": cfg[4],
            "scheduler": cfg[5],
            "resolution": cfg[8],
            "epochs": cfg[9],
            "n_runs": best["_n"],
            "degenerate": best["_degenerate"],
            **{f"{m}_mean": best[f"{m}_mean"] for m in METRICS},
            **{f"{m}_std": best[f"{m}_std"] for m in METRICS},
            "kept_source": best["source_path"],
            "n_duplicates": len(collapsed),
            "collapsed_groups": ";".join(sorted(c["experiment_group"] for c in collapsed)),
        })

    # Sort: domain, dataset, then best weighted-F1 first.
    organised.sort(key=lambda r: (
        DOMAIN_ORDER.get(r["domain"], 9),
        r["dataset"],
        -(to_float(r["weighted_f1_mean"]) or -1),
    ))

    OUT_DIR.mkdir(exist_ok=True)

    # ── all_configs.csv ──────────────────────────────────────────────────────
    cols = ["domain", "dataset", "config", "K", "attention_encoding", "scale",
            "scheduler", "resolution", "epochs", "n_runs", "degenerate"]
    for m in METRICS:
        cols += [f"{m}_mean", f"{m}_std"]
    cols += ["kept_source", "n_duplicates", "collapsed_groups"]
    with (OUT_DIR / "all_configs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(organised)

    # ── best_per_dataset.csv ─────────────────────────────────────────────────
    best_per_ds: dict[str, dict] = {}
    for r in organised:
        if r["degenerate"]:
            continue
        cur = best_per_ds.get(r["dataset"])
        if cur is None or (to_float(r["weighted_f1_mean"]) or -1) > (to_float(cur["weighted_f1_mean"]) or -1):
            best_per_ds[r["dataset"]] = r
    best_rows = sorted(best_per_ds.values(),
                       key=lambda r: (DOMAIN_ORDER.get(r["domain"], 9), r["dataset"]))
    with (OUT_DIR / "best_per_dataset.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(best_rows)

    # ── README.md report ─────────────────────────────────────────────────────
    def pct(mean, std):
        m = to_float(mean)
        if m is None:
            return "--"
        s = to_float(std)
        return f"{m*100:.2f}" + (f" ± {s*100:.2f}" if s else "")

    lines = ["# Organised Results", "",
             f"Deduplicated from `results_summary.csv` "
             f"({len(rows)} raw runs across {len({r['experiment_group'] for r in rows})} groups "
             f"-> {len(organised)} unique configs). "
             f"Duplicate configs collapsed to the best instance (highest weighted-F1, "
             f"then most seeds). Degenerate/collapsed runs (weighted-F1 < "
             f"{DEGENERATE_WF1:.2f}) are flagged and excluded from the best-per-dataset table.",
             "", "## Best configuration per dataset", "",
             "| Domain | Dataset | Best config | Seeds | Acc (%) | W-F1 (%) | M-F1 (%) | AUC (%) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in best_rows:
        lines.append(
            f"| {r['domain']} | {DATASET_PRETTY.get(r['dataset'], r['dataset'])} "
            f"| {r['config']} | {r['n_runs']} "
            f"| {pct(r['weighted_accuracy_mean'], r['weighted_accuracy_std'])} "
            f"| {pct(r['weighted_f1_mean'], r['weighted_f1_std'])} "
            f"| {pct(r['macro_f1_mean'], r['macro_f1_std'])} "
            f"| {pct(r['mean_auc_mean'], r['mean_auc_std'])} |")

    degen = [r for r in organised if r["degenerate"]]
    lines += ["", "## Degenerate runs excluded (training collapse)", ""]
    if degen:
        lines.append("| Dataset | Config | W-F1 (%) | Source |")
        lines.append("|---|---|---|---|")
        for r in degen:
            lines.append(f"| {DATASET_PRETTY.get(r['dataset'], r['dataset'])} | {r['config']} "
                         f"| {pct(r['weighted_f1_mean'], r['weighted_f1_std'])} | `{r['kept_source']}` |")
    else:
        lines.append("None.")

    lines += ["", "## All unique configurations", "",
              "Full deduplicated table in `all_configs.csv` "
              "(one row per dataset x config; `kept_source` gives the winning "
              "experiment folder, `collapsed_groups` lists the duplicates it "
              "superseded).", ""]

    # Per-dataset config counts.
    per_ds = defaultdict(int)
    for r in organised:
        per_ds[r["dataset"]] += 1
    lines.append("| Dataset | Unique configs |")
    lines.append("|---|---|")
    for ds in sorted(per_ds, key=lambda d: (DOMAIN_ORDER.get(DOMAIN.get(d, "Other"), 9), d)):
        lines.append(f"| {DATASET_PRETTY.get(ds, ds)} | {per_ds[ds]} |")

    # ── Experiment-group inventory (raw folders on disk) ─────────────────────
    grp_runs = defaultdict(int)
    for r in rows:
        grp_runs[r["experiment_group"]] += 1
    disk_groups = sorted(p.name for p in ROOT.glob("results*")
                         if p.is_dir() and p.name != OUT_DIR.name)
    lines += ["", "## Experiment-group inventory (raw folders, not moved)", "",
              "| Group folder | Runs (summary.json) |", "|---|---|"]
    for g in disk_groups:
        n = grp_runs.get(g, 0)
        note = "  *(empty -- safe to delete)*" if n == 0 else ""
        lines.append(f"| `{g}/` | {n}{note} |")
    lines += ["",
              "Nothing above was moved or deleted; the raw folders remain the "
              "source of truth and `kept_source` in `all_configs.csv` points "
              "into them.", ""]

    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"Raw runs            : {len(rows)}")
    print(f"Unique configs      : {len(organised)}")
    print(f"Duplicates collapsed: {sum(r['n_duplicates'] for r in organised)}")
    print(f"Degenerate flagged  : {len(degen)}")
    print(f"Best-per-dataset    : {len(best_rows)} datasets")
    print(f"Written to          : {OUT_DIR.relative_to(ROOT)}/  "
          f"(all_configs.csv, best_per_dataset.csv, README.md)")


if __name__ == "__main__":
    main()
