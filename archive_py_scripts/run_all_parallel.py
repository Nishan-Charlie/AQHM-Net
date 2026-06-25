"""
run_all_parallel.py
-------------------
Train ALL datasets for a full 300 epochs with NO early stopping, running
several experiments concurrently to keep the (otherwise under-utilised) GPU busy.

  * 300 epochs, early stopping disabled (patience >> n_epochs; best-val
    checkpoint is still saved and used for evaluation).
  * Single seed (n_runs=1), K=1 quantum heads (canonical model).
  * Concurrency pool of POOL processes (default 3 — safe on an 8 GB GPU at
    ~2 GB/run). Jobs are launched longest-first so the long pole (PathMNIST)
    overlaps the rest, minimising wall-clock makespan.

Each job logs to results_300ep/<tag>.log. One failure does not abort the rest.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime

PY  = r"C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT = "./results_300ep"
POOL = 3   # concurrent processes; 3 ≈ 6 GB on the 8 GB RTX 4070 Laptop

COMMON = [
    "--n_runs", "1",
    "--n_epochs", "300",
    "--patience", "1000",        # >> 300  => early stopping never triggers
    "--n_quantum_heads", "1",
    "--output_dir", OUT,
]

# (tag, dataset-specific args) — ordered longest-running first
JOBS = [
    ("pathmnist",      ["--dataset", "pathmnist", "--contrastive"]),
    ("mnist",          ["--dataset", "mnist"]),
    ("cifar10",        ["--dataset", "cifar10"]),
    ("cifar100",       ["--dataset", "cifar100"]),
    ("mnist_0123",     ["--dataset", "mnist", "--mnist_classes", "0", "1", "2", "3"]),
    ("dermamnist",     ["--dataset", "dermamnist", "--use_focal_loss", "--use_balanced_sampler"]),
    ("pneumoniamnist", ["--dataset", "pneumoniamnist"]),
    ("breastmnist",    ["--dataset", "breastmnist", "--use_focal_loss", "--use_balanced_sampler"]),
]


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None,
                        help="Specific device to use (e.g., 'cuda:0', 'cuda:1', 'cpu').")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pending = list(JOBS)
    running: dict = {}   # proc -> (tag, start_time, logfile)
    done: list = []

    print(f"[{ts()}] launching {len(pending)} datasets, pool={POOL}, "
          f"300 epochs, no early stopping, K=1", flush=True)

    while pending or running:
        # fill the pool
        while pending and len(running) < POOL:
            tag, extra = pending.pop(0)
            lf = open(os.path.join(OUT, f"{tag}.log"), "w")
            
            cmd = [PY, "run_experiment.py"] + extra + COMMON
            if args.device:
                cmd.extend(["--device", args.device])
                
            p = subprocess.Popen(
                cmd,
                stdout=lf, stderr=subprocess.STDOUT,
            )
            running[p] = (tag, time.time(), lf)
            print(f"[{ts()}] START  {tag}  (running: {[r[0] for r in running.values()]})",
                  flush=True)

        # poll for any finished process
        time.sleep(10)
        for p in list(running):
            if p.poll() is not None:
                tag, t0, lf = running.pop(p)
                lf.close()
                mins = (time.time() - t0) / 60.0
                status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
                print(f"[{ts()}] DONE   {tag}  {status}  ({mins:.1f} min)", flush=True)
                done.append((tag, status, mins))

    print(f"\n[{ts()}] ALL COMPLETE")
    for tag, status, mins in done:
        print(f"   {tag:16s} {status:14s} {mins:6.1f} min")


if __name__ == "__main__":
    main()
