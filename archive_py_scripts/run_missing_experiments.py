import os
import subprocess
import time
import sys
from datetime import datetime

PY = "/home/e21283/miniconda3/envs/quantum/bin/python"
OUT_DIR = "./results_combinations"
DEVICE = "cuda:1"
POOL_SIZE = 2  # 2 concurrent jobs to fit comfortably in available GPU memory (~7 GiB free)

# Define standard combination sweeps (K=4, K=8, Scale=small/medium/large) from run_k4_k8_combinations.sh
datasets = ["cifar10", "cifar100", "mnist", "pathmnist", "dermamnist", "pneumoniamnist", "breastmnist"]

jobs_to_run = []

# 1. K=4 and K=8 sweeps from run_k4_k8_combinations.sh
for ds in datasets:
    for k in [4, 8]:
        for scale in ["small", "medium", "large"]:
            # Tag builder matching run_experiment.py
            tag = f"{ds}_k{k}"
            if scale != "small":
                tag += f"_{scale}"
            
            summary_path = os.path.join(OUT_DIR, tag, "summary.json")
            if not os.path.exists(summary_path):
                cmd_args = [
                    "--dataset", ds,
                    "--n_epochs", "200",
                    "--n_runs", "1",
                    "--n_quantum_heads", str(k),
                    "--scale", scale,
                    "--batch_size", "64",
                    "--num_workers", "2",
                    "--output_dir", OUT_DIR,
                    "--device", DEVICE,
                    "--resume"
                ]
                jobs_to_run.append((tag, cmd_args))

# 2. Advanced Preprocessing Sweeps (from run_randaugment_experiments.sh)
randaug_datasets = ["fashionmnist", "mnist", "cifar10"]
for ds in randaug_datasets:
    # 2.1 Full model (K=4, attenc, RandAugment, CutMix, OneCycleLR)
    tag_full = f"{ds}_k4_attenc_micro_onecycle_cutmix1.0_mixup0.8_ra2_9"
    summary_full = os.path.join(OUT_DIR, tag_full, "summary.json")
    if not os.path.exists(summary_full):
        cmd_args = [
            "--dataset", ds,
            "--n_epochs", "100",
            "--n_runs", "1",
            "--scale", "micro",
            "--n_quantum_heads", "4",
            "--attention_encoding",
            "--scheduler", "onecycle",
            "--cutmix_alpha", "1.0",
            "--mixup_alpha", "0.8",
            "--use_randaugment",
            "--batch_size", "64",
            "--num_workers", "2",
            "--output_dir", OUT_DIR,
            "--device", DEVICE,
            "--resume"
        ]
        # Only add contrastive flag for multi-channel RGB datasets
        if ds in ["cifar10", "cifar100", "pathmnist", "dermamnist"]:
            cmd_args.append("--contrastive")
        jobs_to_run.append((tag_full, cmd_args))
        
    # 2.2 No-Quantum Ablation (ablation no_quantum, RandAugment, CutMix, OneCycleLR)
    tag_ab = f"{ds}_no_quantum_micro_onecycle_cutmix1.0_mixup0.8_ra2_9"
    summary_ab = os.path.join(OUT_DIR, tag_ab, "summary.json")
    if not os.path.exists(summary_ab):
        cmd_args = [
            "--dataset", ds,
            "--n_epochs", "100",
            "--n_runs", "1",
            "--scale", "micro",
            "--ablation", "no_quantum",
            "--scheduler", "onecycle",
            "--cutmix_alpha", "1.0",
            "--mixup_alpha", "0.8",
            "--use_randaugment",
            "--batch_size", "64",
            "--num_workers", "2",
            "--output_dir", OUT_DIR,
            "--device", DEVICE,
            "--resume"
        ]
        jobs_to_run.append((tag_ab, cmd_args))

# 3. Micro Canonical Baselines (K=4 and K=4 attenc)
for ds in datasets:
    # 3.1 K=4 attenc micro baseline
    tag_attenc = f"{ds}_k4_attenc_micro"
    summary_attenc = os.path.join(OUT_DIR, tag_attenc, "summary.json")
    if not os.path.exists(summary_attenc):
        cmd_args = [
            "--dataset", ds,
            "--n_epochs", "100",
            "--n_runs", "1",
            "--scale", "micro",
            "--n_quantum_heads", "4",
            "--attention_encoding",
            "--batch_size", "64",
            "--num_workers", "2",
            "--output_dir", OUT_DIR,
            "--device", DEVICE,
            "--resume"
        ]
        jobs_to_run.append((tag_attenc, cmd_args))
        
    # 3.2 K=4 no_quantum micro baseline
    tag_nq = f"{ds}_no_quantum_micro"
    summary_nq = os.path.join(OUT_DIR, tag_nq, "summary.json")
    if not os.path.exists(summary_nq):
        cmd_args = [
            "--dataset", ds,
            "--n_epochs", "100",
            "--n_runs", "1",
            "--scale", "micro",
            "--ablation", "no_quantum",
            "--batch_size", "64",
            "--num_workers", "2",
            "--output_dir", OUT_DIR,
            "--device", DEVICE,
            "--resume"
        ]
        jobs_to_run.append((tag_nq, cmd_args))

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    print(f"[{ts()}] Audited workspace. Found {len(jobs_to_run)} missing or incomplete experiments.")
    if not jobs_to_run:
        print(f"[{ts()}] All experiments are completed! Nothing to run.")
        return

    print(f"[{ts()}] Queue to execute:")
    for tag, _ in jobs_to_run:
        print(f"  - {tag}")
    
    os.makedirs("logs_combinations", exist_ok=True)
    pending = list(jobs_to_run)
    running = {}
    done = []

    print(f"\n[{ts()}] Starting parallel execution loop (pool size = {POOL_SIZE})...")

    while pending or running:
        # Fill the pool
        while pending and len(running) < POOL_SIZE:
            tag, extra_args = pending.pop(0)
            log_file_path = os.path.join("logs_combinations", f"log_{tag}.txt")
            lf = open(log_file_path, "w")
            
            cmd = [PY, "run_experiment.py"] + extra_args
            p = subprocess.Popen(
                cmd,
                stdout=lf, stderr=subprocess.STDOUT,
            )
            running[p] = (tag, time.time(), lf)
            print(f"[{ts()}] LAUNCHED: {tag} (PID: {p.pid}) | Active in pool: {[r[0] for r in running.values()]}")

        # Poll running processes
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                tag, t0, lf = running.pop(p)
                lf.close()
                elapsed_mins = (time.time() - t0) / 60.0
                status = "SUCCESS" if p.returncode == 0 else f"FAILED (code {p.returncode})"
                print(f"[{ts()}] FINISHED: {tag} -> {status} ({elapsed_mins:.2f} min)")
                done.append((tag, status, elapsed_mins))

    print(f"\n[{ts()}] ALL JOBS COMPLETED.")
    for tag, status, mins in done:
        print(f"  {tag:50s} : {status:12s} ({mins:.2f} mins)")

if __name__ == "__main__":
    main()
