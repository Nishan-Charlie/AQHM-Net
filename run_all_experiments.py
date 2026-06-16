"""
run_all_experiments.py
----------------------
Sequential launcher for all AQHM-Net experiments.

Runs:
  1. MNIST-4     (classes 0,1,2,3  — fastest, comparable to Wu et al.)
  2. MNIST-10    (all 10 classes)
  3. PneumoniaMNIST  (binary, grayscale)
  4. BreastMNIST     (binary, grayscale — smallest dataset)
  5. DermaMNIST      (7-class RGB  + contrastive)
  6. PathMNIST       (9-class RGB  + contrastive)
  7. CIFAR-10        (10-class natural images, 32×32 resized to 28×28)
  8. CIFAR-100       (100-class natural images, 32×32 resized to 28×28)

Output structure:
  results/
  ├── mnist_0123/          <- MNIST-4
  ├── mnist/               <- MNIST-10
  ├── pneumoniamnist/
  ├── breastmnist/
  ├── dermamnist/
  ├── pathmnist/
  ├── cifar10/
  ├── cifar100/
  ├── all_datasets_comparison.png
  └── experiment_log.txt

Usage:
    conda run -n pmqnet python run_all_experiments.py
    conda run -n pmqnet python run_all_experiments.py --quick   # 1 run / 5 epochs
    conda run -n pmqnet python run_all_experiments.py --full    # 3 runs / 20 epochs
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SCRIPT      = os.path.join(os.path.dirname(__file__), "run_experiment.py")
# Use the pmqnet env Python directly to avoid conda run activation overhead
_PMQNET_PY  = r"C:\Users\nisha\miniconda3\envs\pmqnet\python.exe"
PYTHON      = _PMQNET_PY if os.path.exists(_PMQNET_PY) else sys.executable


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

def get_experiments(n_runs: int, n_epochs: int) -> list[dict]:
    # Per-dataset timeout (seconds) based on measured epoch times:
    #   MNIST-4      ~30s/ep  × 10 runs × 50 ep =  15 000s → 2.5h cap  (7 200s)
    #   MNIST-10     ~75s/ep  × 10 runs × 50 ep =  37 500s → 6h cap    (21 600s)
    #   Pneumonia    ~7s/ep                                → 1h cap     (3 600s)
    #   Breast       ~1s/ep                                → 0.5h cap   (1 800s)
    #   Derma        ~11s/ep                               → 2h cap     (7 200s)
    #   PathMNIST    ~134s/ep × 10 runs × 50 ep = 67 000s → 12h cap    (43 200s)
    return [
        # ── MNIST-4 (fast, comparable to Wu et al. 2025) ──────────────────
        dict(
            name        = "MNIST-4",
            dataset_tag = "mnist_0123",
            cmd_extra   = ["--dataset", "mnist",
                           "--mnist_classes", "0", "1", "2", "3"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = False,
            timeout     = 14_400,    # 4h
        ),
        # ── MNIST-10 (all digit classes) ───────────────────────────────────
        dict(
            name        = "MNIST-10",
            dataset_tag = "mnist",
            cmd_extra   = ["--dataset", "mnist"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = False,
            timeout     = 25_200,    # 7h
        ),
        # ── PneumoniaMNIST (binary, chest X-ray, grayscale) ───────────────
        dict(
            name        = "PneumoniaMNIST",
            dataset_tag = "pneumoniamnist",
            cmd_extra   = ["--dataset", "pneumoniamnist"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = False,
            timeout     = 5_400,     # 1.5h
        ),
        # ── BreastMNIST (binary, ultrasound, grayscale, 546 train) ────────
        # WeightedRandomSampler prevents mode-collapse on the malignant class;
        # FocalLoss focuses on hard examples in the now-balanced batches.
        dict(
            name        = "BreastMNIST",
            dataset_tag = "breastmnist",
            cmd_extra   = ["--dataset", "breastmnist",
                           "--use_focal_loss", "--use_balanced_sampler"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = False,
            timeout     = 3_600,     # 1h
        ),
        # ── DermaMNIST (7-class RGB dermoscopy) ───────────────────────────
        # Contrastive removed: NT-Xent is ineffective with <2 minority-class
        # samples per batch (df=1.2%, vasc=1.4%). WeightedRandomSampler gives
        # ~18 samples/class/batch; FocalLoss handles residual hard examples.
        dict(
            name        = "DermaMNIST",
            dataset_tag = "dermamnist",
            cmd_extra   = ["--dataset", "dermamnist",
                           "--use_focal_loss", "--use_balanced_sampler"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = False,
            timeout     = 7_200,     # 2h
        ),
        # ── PathMNIST (9-class RGB histology, 90k train) ──────────────────
        dict(
            name        = "PathMNIST",
            dataset_tag = "pathmnist",
            cmd_extra   = ["--dataset", "pathmnist", "--contrastive"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = True,
            timeout     = 43_200,    # 12h (704 batches × ~134s/ep × 10 runs)
        ),
        # ── CIFAR-10 (10-class natural images, 45k train after 90/10 split) ──
        # 32×32 RGB resized to 28×28 to match backbone stride-2 downsampling.
        # Balanced dataset — no sampler or focal loss needed.
        # Measured: ~70s/ep steady-state (351 batches) → 10×50×70=35 000s cap 14h
        dict(
            name        = "CIFAR-10",
            dataset_tag = "cifar10",
            cmd_extra   = ["--dataset", "cifar10"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = False,
            timeout     = 50_400,    # 14h (70s/ep × 10 runs × 50 ep + buffer)
        ),
        # ── CIFAR-100 (100-class natural images, 45k train after 90/10 split) ─
        # Same 351 batches/ep as CIFAR-10; slightly slower due to 100-way CE.
        dict(
            name        = "CIFAR-100",
            dataset_tag = "cifar100",
            cmd_extra   = ["--dataset", "cifar100"],
            n_runs      = n_runs,
            n_epochs    = n_epochs,
            contrastive = False,
            timeout     = 57_600,    # 16h buffer
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, logfile) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    logfile.write(line + "\n")
    logfile.flush()


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def load_summary(results_dir: str, dataset_tag: str) -> dict | None:
    path = os.path.join(results_dir, dataset_tag, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(args) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "experiment_log.txt")

    experiments = get_experiments(args.n_runs, args.n_epochs)
    total = len(experiments)

    with open(log_path, "w") as logfile:
        log(f"AQHM-Net Experiment Suite", logfile)
        log(f"  n_runs={args.n_runs}  n_epochs={args.n_epochs}  batch={args.batch_size}", logfile)
        log(f"  datasets ({total}): {[e['name'] for e in experiments]}", logfile)
        log(f"  output  : {RESULTS_DIR}", logfile)
        log("=" * 64, logfile)

        suite_start = time.time()
        completed   = []
        failed      = []

        for i, exp in enumerate(experiments, 1):
            log(f"\n[{i}/{total}] Starting: {exp['name']}", logfile)
            t0 = time.time()

            cmd = [
                PYTHON, "-u", SCRIPT,   # -u: unbuffered stdout
                "--output_dir", RESULTS_DIR,
                "--data_root",  os.path.join(os.path.dirname(__file__), "data"),
                "--batch_size", str(args.batch_size),
                "--n_runs",     str(exp["n_runs"]),
                "--n_epochs",   str(exp["n_epochs"]),
                "--num_workers", "0",   # avoid multiprocessing issues on Windows
                "--resume",             # skip already-completed seeds on restart
            ] + exp["cmd_extra"]

            log(f"  cmd: {' '.join(cmd[2:])}", logfile)  # skip python + script path

            try:
                subprocess.run(
                    cmd,
                    capture_output=False,   # let stdout/stderr stream to terminal
                    check=True,
                    timeout=exp["timeout"],
                )
                elapsed = time.time() - t0
                log(f"  DONE in {format_duration(elapsed)}", logfile)
                completed.append(exp["name"])

                # Read and log the summary
                summary = load_summary(RESULTS_DIR, exp["dataset_tag"])
                if summary:
                    wf1 = summary.get("weighted_f1", {})
                    wacc = summary.get("weighted_accuracy", {})
                    log(
                        f"  W-F1  : {wf1.get('mean', 0):.4f} ± {wf1.get('std', 0):.4f}",
                        logfile,
                    )
                    log(
                        f"  W-Acc : {wacc.get('mean', 0):.4f} ± {wacc.get('std', 0):.4f}",
                        logfile,
                    )

            except subprocess.CalledProcessError as e:
                elapsed = time.time() - t0
                log(f"  FAILED (exit {e.returncode}) after {format_duration(elapsed)}", logfile)
                failed.append(exp["name"])

            except subprocess.TimeoutExpired:
                h_cap = exp["timeout"] // 3600
                log(f"  TIMEOUT (>{h_cap}h cap) — skipping", logfile)
                failed.append(exp["name"])

        # ── Final summary ────────────────────────────────────────────────────
        total_time = time.time() - suite_start
        log("\n" + "=" * 64, logfile)
        log(f"SUITE COMPLETE in {format_duration(total_time)}", logfile)
        log(f"  Completed : {len(completed)}/{total}  — {completed}", logfile)
        if failed:
            log(f"  Failed    : {failed}", logfile)

        # Collect all summaries for final comparison table
        log("\nFINAL RESULTS TABLE:", logfile)
        log(f"{'Dataset':<20}  {'W-F1':>10}  {'±':>6}  {'W-Acc':>10}  {'±':>6}  {'mAUC':>10}  {'±':>6}", logfile)
        log("-" * 78, logfile)

        for exp in experiments:
            s = load_summary(RESULTS_DIR, exp["dataset_tag"])
            if s is None:
                log(f"  {exp['name']:<18}  {'—':>10}", logfile)
                continue
            wf1  = s.get("weighted_f1",       {})
            wacc = s.get("weighted_accuracy",  {})
            mauc = s.get("mean_auc",           {})
            log(
                f"  {exp['name']:<18}  "
                f"{wf1.get('mean',0):>10.4f}  {wf1.get('std',0):>6.4f}  "
                f"{wacc.get('mean',0):>10.4f}  {wacc.get('std',0):>6.4f}  "
                f"{mauc.get('mean',0):>10.4f}  {mauc.get('std',0):>6.4f}",
                logfile,
            )

        log(f"\nLog saved -> {log_path}", logfile)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all AQHM-Net experiments sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast sanity check: 1 run x 3 epochs x batch=128.",
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Full paper protocol: 10 runs x 50 epochs x batch=128. (~5-6 hr on GPU)",
    )
    parser.add_argument("--n_runs",     type=int, default=None)
    parser.add_argument("--n_epochs",   type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    # Resolve preset modes
    if args.quick:
        args.n_runs   = args.n_runs   or 1
        args.n_epochs = args.n_epochs or 3
    elif args.paper:
        args.n_runs   = args.n_runs   or 10
        args.n_epochs = args.n_epochs or 50
    else:
        # Default: intermediate check (3 runs x 20 epochs)
        args.n_runs   = args.n_runs   or 3
        args.n_epochs = args.n_epochs or 20

    run_all(args)
