"""
run_experiment.py
-----------------
Top-level entry point for AQHM-Net experiments.

Output structure (one directory per dataset):

    results/
    └── {dataset}/
        ├── checkpoints/
        │   └── aqhm_net_seed_000_best.pt   ← best model weights
        ├── histories/
        │   └── aqhm_net_seed_000_history.json
        ├── metrics/
        │   └── seed_000_test_metrics.json
        ├── plots/
        │   ├── training_curves.png
        │   ├── confusion_matrix.png
        │   ├── per_class_auc.png
        │   ├── metrics_boxplot.png
        │   └── loss_vs_seed.png
        └── summary.json                    ← mean ± std across all seeds
    results/
    └── all_datasets_comparison.png         ← written after ALL datasets run

Usage examples:

    # Quick sanity check (1 epoch, 1 seed, batch=8)
    python run_experiment.py --dataset mnist --debug

    # MNIST-4
    python run_experiment.py --dataset mnist --mnist_classes 0 1 2 3 --n_epochs 50 --n_runs 10

    # FashionMNIST
    python run_experiment.py --dataset fashionmnist --n_epochs 50 --n_runs 10

    # PathMNIST with contrastive loss
    python run_experiment.py --dataset pathmnist --contrastive --n_epochs 50 --n_runs 10

    # Ablation — no UIB backbone
    python run_experiment.py --dataset mnist --ablation no_uib --n_epochs 50
"""

import argparse
import json
import os

import numpy as np
import torch

from aqhm_net.dataset import get_dataloaders, DATASET_CONFIG, compute_class_weights
from aqhm_net.model import AQHMNet
from aqhm_net.train import train_model
from aqhm_net.evaluate import evaluate_model, aggregate_runs, print_results_table
from aqhm_net.plotting import save_all_plots, plot_dataset_comparison


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AQHM-Net: Attention-Guided Hybrid Quantum MobileNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset to use.",
    )
    parser.add_argument(
        "--mnist_classes", type=int, nargs="+", default=None, metavar="C",
        help="MNIST digit classes (e.g. 0 1 2 3 for MNIST-4).",
    )
    parser.add_argument(
        "--fashion_classes", type=int, nargs="+", default=None, metavar="C",
        help="FashionMNIST garment indices (0-9). E.g. 0 6 for T-shirt vs Shirt.",
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for dataset downloads.",
    )

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--n_epochs",   type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--n_runs",     type=int,   default=10,
                        help="Number of independent runs (different seeds).")
    parser.add_argument("--base_seed",  type=int,   default=0)
    parser.add_argument("--patience",   type=int,   default=10,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--contrastive", action="store_true",
                        help="Enable NT-Xent contrastive alignment (for RGB datasets).")
    parser.add_argument("--contrastive_weight", type=float, default=0.15)

    # ── Ablation ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=["no_uib", "z_basis"],
        help="Which ablation variant to run (None = full model).",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Root directory for all outputs (checkpoints, histories, plots).",
    )
    parser.add_argument("--num_workers", type=int, default=0)

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 epoch, 1 run, batch=8.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip seeds that already have a completed history file "
                             "and evaluate ALL completed seeds at the end.")

    # ── Loss / sampling improvements for imbalanced medical datasets ─────────
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Focal Loss instead of CrossEntropyLoss.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal Loss focusing parameter gamma (default 2.0).")
    parser.add_argument("--use_balanced_sampler", action="store_true",
                        help="Use WeightedRandomSampler for balanced mini-batches.")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation even if the dataset "
                             "config defines tta_views (e.g. for ablation).")
    parser.add_argument("--no_clahe", action="store_true",
                        help="Disable CLAHE preprocessing (grayscale medical ablation).")
    parser.add_argument("--no_color_constancy", action="store_true",
                        help="Disable Shades-of-Gray colour constancy (RGB medical ablation).")
    parser.add_argument("--n_quantum_heads", type=int, default=1,
                        help="K parallel quantum circuits over patch groups "
                             "(optional). K=1 = original single circuit; ablate "
                             "over {1,2,4,8}. Output dir is tagged _kK when K>1.")
    parser.add_argument("--attention_encoding", action="store_true",
                        help="Attention-conditioned trainable quantum encoding "
                             "(SSA weights pool patches + learned angle map) in "
                             "place of fixed arctan(mean). Tags output dir _attenc.")
    parser.add_argument("--img_size", type=int, default=28,
                        help="MedMNIST native resolution {28,64,128,224}. The "
                             "resolution-adaptive backbone pools to 7x7. Tags _rN.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output directory builder
# ---------------------------------------------------------------------------

def make_output_dirs(base: str, dataset: str) -> dict[str, str]:
    """Create and return all subdirectories for one dataset experiment."""
    root = os.path.join(base, dataset)
    dirs = {
        "root":        root,
        "checkpoints": os.path.join(root, "checkpoints"),
        "histories":   os.path.join(root, "histories"),
        "metrics":     os.path.join(root, "metrics"),
        "plots":       os.path.join(root, "plots"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    args: argparse.Namespace,
    in_channels: int,
    num_classes: int,
) -> AQHMNet:
    if args.ablation == "no_uib":
        return AQHMNet.ablation_no_uib(in_channels, num_classes)
    elif args.ablation == "z_basis":
        return AQHMNet.ablation_z_basis(in_channels, num_classes)
    else:
        return AQHMNet(
            in_channels=in_channels,
            num_classes=num_classes,
            use_contrastive=args.contrastive,
            contrastive_weight=args.contrastive_weight,
            n_quantum_heads=args.n_quantum_heads,
            attention_encoding=args.attention_encoding,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Debug overrides
    if args.debug:
        args.n_runs    = 1
        args.n_epochs  = 1
        args.batch_size = 8
        debug_batches   = 3   # only 3 batches per epoch (24 samples total)
        print("[DEBUG] 1 run / 1 epoch / batch_size=8 / 3 batches")
    else:
        debug_batches = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*64}")
    print(f"  AQHM-Net Experiment Runner")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Ablation : {args.ablation or 'None (full model)'}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.n_epochs}   Runs: {args.n_runs}")
    print(f"{'='*64}\n")

    # ── Output directories ───────────────────────────────────────────────────
    # Append ablation tag so ablation results never overwrite full-model results
    dataset_tag = args.dataset
    if args.ablation:
        dataset_tag += f"_{args.ablation}"
    if args.mnist_classes and args.dataset == "mnist":
        dataset_tag += "_" + "".join(map(str, args.mnist_classes))
    if args.n_quantum_heads > 1:
        dataset_tag += f"_k{args.n_quantum_heads}"
    if args.attention_encoding:
        dataset_tag += "_attenc"
    if args.img_size != 28:
        dataset_tag += f"_r{args.img_size}"

    dirs = make_output_dirs(args.output_dir, dataset_tag)
    print(f"[output] Results -> {dirs['root']}")

    # ── Preprocessing ablation overrides (mutate config before loaders build) ──
    _cfg = DATASET_CONFIG[args.dataset.lower()]
    if args.no_clahe and _cfg.get("clahe"):
        _cfg["clahe"] = False
        print("[preproc] CLAHE disabled via --no_clahe")
    if args.no_color_constancy and _cfg.get("color_constancy"):
        _cfg["color_constancy"] = False
        print("[preproc] colour constancy disabled via --no_color_constancy")

    # ── Data ─────────────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mnist_classes=args.mnist_classes,
        fashion_classes=args.fashion_classes,
        seed=args.base_seed,
        use_balanced_sampler=args.use_balanced_sampler,
        img_size=args.img_size,
    )

    cfg = DATASET_CONFIG[args.dataset.lower()]
    in_channels = cfg["channels"]
    num_classes  = cfg["num_classes"]

    if args.dataset == "mnist" and args.mnist_classes is not None:
        num_classes = len(args.mnist_classes)
    if args.dataset == "fashionmnist" and args.fashion_classes is not None:
        num_classes = len(args.fashion_classes)

    class_names: list[str] | None = cfg.get("class_names")
    print(f"  Classes : {num_classes}  |  Channels: {in_channels}\n")

    # Class weights for imbalanced datasets.
    # Skip when using WeightedRandomSampler — the sampler already balances
    # the batch distribution, so additional loss weighting is redundant and
    # can cause the same mode-collapse instability we're trying to avoid.
    is_medical = cfg.get("medmnist", False)
    if is_medical and not args.use_balanced_sampler:
        cw = compute_class_weights(loaders["train"], num_classes, device=device)
        print(f"  Class weights: {cw.tolist()}")
    else:
        cw = None

    # ── Multi-run training ───────────────────────────────────────────────────
    all_histories: list[dict] = []

    for run_idx in range(args.n_runs):
        seed = args.base_seed + run_idx
        run_id = f"seed_{seed:03d}"

        dst_hist = os.path.join(dirs["histories"], f"aqhm_net_{run_id}_history.json")

        # --resume: skip seeds with a complete history.
        # A history is "complete" only if it ran long enough to be genuine:
        # early-stopping requires at least patience+1 epochs, so any history
        # with stopped_epoch < patience+1 is from a short quick-run and must
        # be re-trained.
        if args.resume and os.path.exists(dst_hist):
            with open(dst_hist) as f:
                _h = json.load(f)
            _stopped = _h.get("stopped_epoch", len(_h.get("train_loss", [])))
            _min_valid = args.patience + 2   # must have run at least this many epochs
            if _stopped >= _min_valid:
                print(f"\n  [resume] seed {seed} already done ({_stopped} ep) — loading history")
                _h["_run_id"] = run_id
                all_histories.append(_h)
                continue
            else:
                print(f"\n  [resume] seed {seed} history too short ({_stopped} ep < {_min_valid}) — re-training")

        # Seed everything for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n{'-'*60}")
        print(f"  RUN {run_idx + 1}/{args.n_runs}  (seed={seed})")
        print(f"{'-'*60}")

        model = build_model(args, in_channels, num_classes)

        history = train_model(
            model,
            loaders,
            n_epochs=args.n_epochs,
            contrastive_weight=args.contrastive_weight,
            patience=args.patience,
            checkpoint_dir=dirs["checkpoints"],
            run_id=run_id,
            device=device,
            verbose=True,
            debug_batches=debug_batches,
            class_weights=cw,
            label_smoothing=0.10,
            warmup_epochs=5,
            use_focal=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
        )

        # Move the history JSON into histories/
        src_hist = os.path.join(dirs["checkpoints"], f"aqhm_net_{run_id}_history.json")
        if os.path.exists(src_hist):
            os.replace(src_hist, dst_hist)
        history["_run_id"] = run_id
        all_histories.append(history)

    # ── Per-run test evaluation ───────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print("  Evaluating all seeds on test split ...")
    print(f"{'-'*60}")

    per_run_metrics: list[dict] = []

    # Test-time augmentation views from the dataset config (medical sets only;
    # never for MNIST digits). Disabled with --no_tta.
    tta_views = None if args.no_tta else cfg.get("tta_views")
    if tta_views:
        print(f"  [TTA] averaging over views: {tta_views}")

    for run_idx in range(args.n_runs):
        seed   = args.base_seed + run_idx
        run_id = f"seed_{seed:03d}"
        ckpt   = os.path.join(dirs["checkpoints"], f"aqhm_net_{run_id}_best.pt")

        if not os.path.exists(ckpt):
            print(f"  [WARN] checkpoint not found: {ckpt}")
            continue

        model = build_model(args, in_channels, num_classes).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        metrics = evaluate_model(
            model, loaders["test"], device, num_classes, tta_views=tta_views
        )
        metrics["seed"] = seed

        # Save individual seed metrics
        m_path = os.path.join(dirs["metrics"], f"{run_id}_test_metrics.json")
        with open(m_path, "w") as f:
            json.dump(metrics, f, indent=2)

        per_run_metrics.append(metrics)
        print(f"  seed {seed:03d}  W-F1={metrics['weighted_f1']:.4f}  "
              f"W-Acc={metrics['weighted_accuracy']:.4f}  "
              f"mAUC={metrics['mean_auc']:.4f}")

    # ── Print aggregated results ─────────────────────────────────────────────
    model_name = f"AQHM-Net{'-' + args.ablation if args.ablation else ''}"
    print_results_table(per_run_metrics, dataset_tag, model_name)

    # ── Aggregate summary ────────────────────────────────────────────────────
    summary = {}
    for key in ["weighted_f1", "macro_f1", "weighted_accuracy", "mean_auc"]:
        agg = aggregate_runs(per_run_metrics, key)
        summary[key] = agg

    # Identify best seed (lowest best_val_loss)
    best_seed_idx = int(np.argmin(
        [h.get("best_val_loss", 1e9) for h in all_histories]
    ))
    summary["best_seed_idx"] = best_seed_idx
    summary["n_runs"]        = args.n_runs

    summary_path = os.path.join(dirs["root"], "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[output] Summary saved -> {summary_path}")

    # ── Generate all plots ───────────────────────────────────────────────────
    if per_run_metrics:
        save_all_plots(
            histories=all_histories,
            per_run_metrics=per_run_metrics,
            output_dir=dirs["plots"],
            dataset_name=dataset_tag,
            class_names=class_names,
            best_seed_idx=best_seed_idx,
        )

    # ── Multi-dataset comparison (if sibling summaries exist) ────────────────
    _try_update_comparison(args.output_dir, dataset_tag, summary)

    print(f"\n{'='*64}")
    print(f"  Experiment complete!")
    print(f"  Results : {dirs['root']}")
    print(f"  Plots   : {dirs['plots']}")
    print(f"{'='*64}\n")


def _try_update_comparison(results_root: str, current_dataset: str, current_summary: dict) -> None:
    """Regenerate the cross-dataset comparison plot if ≥2 summaries exist."""
    summaries: dict[str, dict] = {}

    for entry in os.scandir(results_root):
        if not entry.is_dir():
            continue
        s_path = os.path.join(entry.path, "summary.json")
        if not os.path.exists(s_path):
            continue
        with open(s_path) as f:
            s = json.load(f)
        # Flatten to top-level mean/std
        flat = {
            "val_acc_mean":      s.get("weighted_accuracy", {}).get("mean", 0.0),
            "val_acc_std":       s.get("weighted_accuracy", {}).get("std",  0.0),
            "weighted_f1_mean":  s.get("weighted_f1",       {}).get("mean", 0.0),
            "weighted_f1_std":   s.get("weighted_f1",       {}).get("std",  0.0),
        }
        summaries[entry.name] = flat

    if len(summaries) >= 2:
        print(f"\n[plots] Updating cross-dataset comparison ({len(summaries)} datasets)...")
        plot_dataset_comparison(summaries, results_root)


if __name__ == "__main__":
    main()
