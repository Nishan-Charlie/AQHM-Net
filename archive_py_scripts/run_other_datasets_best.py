import subprocess
import os
import time

# Create log directory
os.makedirs("logs_combinations", exist_ok=True)

# Path to python environment
python_bin = "/home/e21283/miniconda3/envs/quantum/bin/python"
output_dir = "./results_combinations"

# Other datasets to run
datasets = ["cifar100", "pathmnist", "dermamnist", "pneumoniamnist", "breastmnist"]

running_jobs = []

# Allocate jobs to GPUs (3 on cuda:0, 2 on cuda:1)
gpu_assignments = {
    "cifar100": 0,
    "pathmnist": 0,
    "dermamnist": 0,
    "pneumoniamnist": 1,
    "breastmnist": 1
}

for ds in datasets:
    gpu_id = gpu_assignments[ds]
    
    # Tag for folders and logs
    desc = f"{ds}_k4_attenc_micro_onecycle_cutmix1.0_mixup0.8_ra2_9"
    log_file = f"logs_combinations/log_{desc}.txt"
    
    cmd = [
        python_bin, "run_experiment.py",
        "--dataset", ds,
        "--n_epochs", "200",
        "--n_runs", "1",
        "--scale", "micro",
        "--n_quantum_heads", "4",
        "--attention_encoding",
        "--scheduler", "onecycle",
        "--cutmix_alpha", "1.0",
        "--mixup_alpha", "0.8",
        "--use_randaugment",
        "--output_dir", output_dir,
        "--device", f"cuda:{gpu_id}",
        "--resume"
    ]
    
    # Enable contrastive loss for RGB datasets
    if ds in ["cifar100", "pathmnist", "dermamnist"]:
        cmd.append("--contrastive")
        
    print(f"[LAUNCH] Starting {ds} (Best Config) on cuda:{gpu_id}")
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    running_jobs.append({
        "proc": proc,
        "ds": ds,
        "gpu_id": gpu_id
    })

# Monitor runs
last_status_time = 0
while running_jobs:
    still_running = []
    for job in running_jobs:
        ret = job["proc"].poll()
        if ret is not None:
            if ret == 0:
                print(f"[SUCCESS] {job['ds']} completed successfully.")
            else:
                print(f"[FAILED] {job['ds']} failed with exit code {ret}.")
        else:
            still_running.append(job)
    running_jobs = still_running
    
    current_time = time.time()
    if current_time - last_status_time >= 30 and running_jobs:
        active = [job["ds"] for job in running_jobs]
        print(f"Status: Running {len(active)} jobs {active}...")
        last_status_time = current_time
        
    time.sleep(5)

print("All other datasets best config runs completed.")
