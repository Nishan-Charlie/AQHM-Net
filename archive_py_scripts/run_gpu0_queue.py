import subprocess
import time
import os
from datetime import datetime

python_bin = "/home/e21283/miniconda3/envs/quantum/bin/python"
output_dir = "./results_combinations"

# Define the sequential queue of jobs to execute on GPU 0 (cuda:0)
jobs = [
    {
        "desc": "dermamnist_k4_attenc_micro_onecycle_standard",
        "log_file": "logs_combinations/log_dermamnist_k4_attenc_micro_onecycle.txt",
        "log_mode": "w",
        "cmd": [
            python_bin, "run_experiment.py",
            "--dataset", "dermamnist",
            "--n_epochs", "200",
            "--n_runs", "1",
            "--scale", "micro",
            "--n_quantum_heads", "4",
            "--attention_encoding",
            "--scheduler", "onecycle",
            "--cutmix_alpha", "0.0",
            "--mixup_alpha", "0.0",
            "--output_dir", output_dir,
            "--device", "cuda:0",
            "--contrastive",
            "--resume"
        ]
    },
    {
        "desc": "cifar10_medium_resumed",
        "log_file": "logs_combinations/log_cifar10_medium_k1_r28_augment.txt",
        "log_mode": "a",
        "cmd": [
            python_bin, "run_experiment.py",
            "--dataset", "cifar10",
            "--n_epochs", "200",
            "--n_runs", "1",
            "--scale", "medium",
            "--n_quantum_heads", "1",
            "--scheduler", "onecycle",
            "--cutmix_alpha", "1.0",
            "--mixup_alpha", "0.8",
            "--use_randaugment",
            "--output_dir", output_dir,
            "--device", "cuda:0",
            "--resume"
        ]
    },
    {
        "desc": "cifar10_k4_medium_resumed",
        "log_file": "logs_combinations/log_cifar10_medium_k4_r28_augment.txt",
        "log_mode": "a",
        "cmd": [
            python_bin, "run_experiment.py",
            "--dataset", "cifar10",
            "--n_epochs", "200",
            "--n_runs", "1",
            "--scale", "medium",
            "--n_quantum_heads", "4",
            "--scheduler", "onecycle",
            "--cutmix_alpha", "1.0",
            "--mixup_alpha", "0.8",
            "--use_randaugment",
            "--output_dir", output_dir,
            "--device", "cuda:0",
            "--resume"
        ]
    },
    {
        "desc": "cifar10_k8_medium_resumed",
        "log_file": "logs_combinations/log_cifar10_medium_k8_r28_augment.txt",
        "log_mode": "a",
        "cmd": [
            python_bin, "run_experiment.py",
            "--dataset", "cifar10",
            "--n_epochs", "200",
            "--n_runs", "1",
            "--scale", "medium",
            "--n_quantum_heads", "8",
            "--scheduler", "onecycle",
            "--cutmix_alpha", "1.0",
            "--mixup_alpha", "0.8",
            "--use_randaugment",
            "--output_dir", output_dir,
            "--device", "cuda:0",
            "--resume"
        ]
    }
]

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{ts()}] Starting run_gpu0_queue.py. Running {len(jobs)} jobs sequentially on GPU 0...")

for i, job in enumerate(jobs):
    print(f"[{ts()}] Job {i+1}/{len(jobs)}: Starting {job['desc']}...")
    print(f"[{ts()}] Command: {' '.join(job['cmd'])}")
    print(f"[{ts()}] Logging to {job['log_file']} (mode: {job['log_mode']})")
    
    # Pre-append header to log if in append mode
    if job['log_mode'] == 'a' and os.path.exists(job['log_file']):
        with open(job['log_file'], "a") as lf:
            lf.write(f"\n\n================ RESUMING JOB ON {ts()} ================\n\n")
            
    t0 = time.time()
    with open(job['log_file'], job['log_mode']) as lf:
        proc = subprocess.Popen(job['cmd'], stdout=lf, stderr=subprocess.STDOUT)
        
    ret = proc.wait()
    elapsed = (time.time() - t0) / 60.0
    
    if ret == 0:
        print(f"[{ts()}] SUCCESS: {job['desc']} completed in {elapsed:.2f} mins.")
    else:
        print(f"[{ts()}] FAILED: {job['desc']} exited with code {ret} after {elapsed:.2f} mins.")

print(f"[{ts()}] All queued jobs completed.")
