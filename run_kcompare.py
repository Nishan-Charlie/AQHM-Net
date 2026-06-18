"""
run_kcompare.py
---------------
Significance experiment: does K=8 (parallel quantum circuits) actually beat
K=1 on CIFAR? CIFAR-10 and CIFAR-100, K in {1, 8}, 5 seeds each, 100 epochs,
early stopping disabled (best-val checkpoint per seed still saved).

4 configs run concurrently (pool=4, ~1.1 GB each ≈ 4.5 GB on the 8 GB GPU).
Each config trains 5 seeds sequentially -> summary.json aggregates them, and
the 5 best checkpoints land in <config>/checkpoints/. Per-seed metrics in
<config>/metrics/ give the paired values for a Wilcoxon test afterwards.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime

PY  = r"C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT = "./results_kcompare"
POOL = 4

COMMON = [
    "--n_runs", "5",          # seeds 0..4
    "--n_epochs", "100",
    "--patience", "1000",     # >> 100 => no early stopping
    "--base_seed", "0",
    "--output_dir", OUT,
]

# tag is just for the log filename; run_experiment appends _k8 to its own dirs
JOBS = [
    ("cifar10_k1",  ["--dataset", "cifar10",  "--n_quantum_heads", "1"]),
    ("cifar10_k8",  ["--dataset", "cifar10",  "--n_quantum_heads", "8"]),
    ("cifar100_k1", ["--dataset", "cifar100", "--n_quantum_heads", "1"]),
    ("cifar100_k8", ["--dataset", "cifar100", "--n_quantum_heads", "8"]),
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
    running: dict = {}
    done: list = []

    print(f"[{ts()}] K-compare: CIFAR-10/100 x K={{1,8}} x 5 seeds x 100ep, "
          f"pool={POOL}, no early stopping", flush=True)

    while pending or running:
        while pending and len(running) < POOL:
            tag, extra = pending.pop(0)
            lf = open(os.path.join(OUT, f"{tag}.log"), "w")
            
            cmd = [PY, "run_experiment.py"] + extra + COMMON
            if args.device:
                cmd.extend(["--device", args.device])
                
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            running[p] = (tag, time.time(), lf)
            print(f"[{ts()}] START  {tag}", flush=True)

        time.sleep(15)
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
        print(f"   {tag:14s} {status:14s} {mins:7.1f} min")


if __name__ == "__main__":
    main()
