"""Experiment launcher — 224-resolution plan.

Design decisions (locked in with the user):
  * Native resolution 224 for every run; NO input/resolution ablations.
  * Attention is ALWAYS ON, except inside the CIFAR-10 attention ablation,
    which is the ONLY place attention is toggled off.
  * The attention ablation runs for CIFAR-10 only, at scales micro/small/medium
    ("Base" == medium).
  * CutMix/MixUp+RandAugment is MANDATORY for CIFAR-10 (not ablated).
  * The 4 MedMNIST datasets get the normal ablation suite
    (full, no_uib, z_basis, no_quantum) at micro scale. CutMix is NOT applied
    to medical data (run_experiment.py also force-disables it for DermaMNIST).

The CIFAR-10 micro + attention-ON run is the modern (224 + CutMix) descendant of
the historical config in which the micro quantum model beat the no-quantum model
(CIFAR-10 micro: quantum K4+attn 0.7874 wF1 vs no_quantum 0.7643 wF1).

Usage:
    python run_experiments_224.py --dry-run     # print the plan, run nothing
    python run_experiments_224.py               # execute sequentially
    python run_experiments_224.py --device cuda:0
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable                  # portable: use the active interpreter
OUTPUT_DIR = "./results_224"
LOG_DIR = ROOT / "logs_224"

N_EPOCHS = "200"
N_RUNS = "1"
IMG = "224"
K = "4"                                  # quantum heads for all quantum variants

MEDICAL_DATASETS = ["dermamnist", "breastmnist", "pneumoniamnist", "pathmnist"]
MEDICAL_ABLATIONS = [None, "no_uib", "z_basis", "no_quantum"]   # None = full model
CIFAR_SCALES = ["micro", "small", "medium"]     # "Base" == medium


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Tags in the "important first" subset (--priority): establishes that attention
# helps on CIFAR-10 micro, the headline MedMNIST full-model numbers, and the
# critical quantum-vs-classical (no_quantum) comparison on every medical set.
# Deferred for later: CIFAR-10 small/medium scale sweep, and the structural
# ablations no_uib / z_basis.
PRIORITY_TAGS = {
    "cifar10_micro_attnon",
    "cifar10_micro_attnoff",
    "dermamnist_micro_full",
    "dermamnist_micro_no_quantum",
    "breastmnist_micro_full",
    "breastmnist_micro_no_quantum",
    "pneumoniamnist_micro_full",
    "pneumoniamnist_micro_no_quantum",
    "pathmnist_micro_full",
    "pathmnist_micro_no_quantum",
}


def build_jobs(device: str | None):
    """Return a list of (tag, cmd) tuples."""
    jobs: list[tuple[str, list[str]]] = []

    def base(dataset: str):
        cmd = [PYTHON, "run_experiment.py",
               "--dataset", dataset,
               "--n_epochs", N_EPOCHS, "--n_runs", N_RUNS,
               "--img_size", IMG,
               "--output_dir", OUTPUT_DIR, "--resume"]
        if device:
            cmd += ["--device", device]
        return cmd

    # ── Group 1: CIFAR-10 attention ablation (micro/small/medium, on vs off) ──
    # CutMix mandatory; onecycle; contrastive; K=4.
    for scale in CIFAR_SCALES:
        for attn_on in (True, False):
            cmd = base("cifar10") + [
                "--scale", scale,
                "--n_quantum_heads", K,
                "--scheduler", "onecycle",
                "--cutmix_alpha", "1.0", "--mixup_alpha", "0.8",
                "--use_randaugment",
                "--contrastive",
            ]
            if attn_on:
                cmd += ["--attention_encoding"]
            tag = f"cifar10_{scale}_attn{'on' if attn_on else 'off'}"
            jobs.append((tag, cmd))

    # ── Group 2: MedMNIST normal ablations (micro, attention on, no CutMix) ──
    for dataset in MEDICAL_DATASETS:
        for abl in MEDICAL_ABLATIONS:
            cmd = base(dataset) + [
                "--scale", "micro",
                "--n_quantum_heads", K,
                "--scheduler", "cosine",
                "--attention_encoding",
                "--contrastive",
            ]
            if abl is not None:
                cmd += ["--ablation", abl]
            tag = f"{dataset}_micro_{abl or 'full'}"
            jobs.append((tag, cmd))

    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit without running anything.")
    ap.add_argument("--priority", action="store_true",
                    help="Run only the important-first subset (see PRIORITY_TAGS).")
    ap.add_argument("--device", type=str, default=None,
                    help="Device passed through to run_experiment.py (e.g. cuda:0).")
    args = ap.parse_args()

    jobs = build_jobs(args.device)
    if args.priority:
        jobs = [(t, c) for (t, c) in jobs if t in PRIORITY_TAGS]

    label = "priority subset" if args.priority else "full plan"
    print(f"{'='*70}\n  224-resolution experiment plan ({label}) — {len(jobs)} jobs\n{'='*70}")
    for i, (tag, cmd) in enumerate(jobs, 1):
        print(f"  [{i:02d}] {tag}")
        print(f"       {' '.join(cmd)}")

    if args.dry_run:
        print("\n[dry-run] Nothing executed.")
        return

    LOG_DIR.mkdir(exist_ok=True)
    print(f"\n[{ts()}] Executing {len(jobs)} jobs sequentially. Logs -> {LOG_DIR}")
    for i, (tag, cmd) in enumerate(jobs, 1):
        log_path = LOG_DIR / f"{i:02d}_{tag}.log"
        print(f"[{ts()}] ({i}/{len(jobs)}) {tag}  ->  {log_path.name}")
        with open(log_path, "a") as fh:
            fh.write(f"\n===== {tag} @ {ts()} =====\n{' '.join(cmd)}\n\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            rc = proc.wait()
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[{ts()}]     {status}")
    print(f"[{ts()}] All jobs finished.")


if __name__ == "__main__":
    main()
