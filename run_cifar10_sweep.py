import subprocess
import time
import sys
import os

# Create log directory
os.makedirs("logs_combinations", exist_ok=True)

# Define configurations to run
scales = ["micro", "small", "medium", "large"]
img_sizes = [28, 224]
ks = [1, 4, 8]

jobs = []
for scale in scales:
    for img_size in img_sizes:
        for k in ks:
            jobs.append({
                "scale": scale,
                "img_size": img_size,
                "k": k
            })

# Path to python environment
python_bin = "/home/e21283/miniconda3/envs/quantum/bin/python"

running_jobs = []

def get_free_memory():
    try:
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        res = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        return [int(x) for x in res]
    except Exception as e:
        print(f"Error querying GPU memory: {e}")
        return [0, 0] # fallback

print(f"Loaded {len(jobs)} jobs in the sweep queue.")

# To prevent flooding the console, track status printing
last_status_time = 0

while jobs or running_jobs:
    # 1. Check on running jobs
    still_running = []
    for job in running_jobs:
        ret = job["proc"].poll()
        if ret is not None:
            # Job finished
            if ret == 0:
                print(f"[SUCCESS] Job Scale:{job['scale']} Size:{job['img_size']} K:{job['k']} completed.")
            else:
                print(f"[FAILED] Job Scale:{job['scale']} Size:{job['img_size']} K:{job['k']} exited with code {ret}.")
        else:
            still_running.append(job)
    running_jobs = still_running

    # 2. Assign new jobs to GPUs if slots and memory are available
    free_mem = get_free_memory()
    
    for gpu_id in [0, 1]:
        # Count jobs running on this GPU
        gpu_running_count = sum(1 for j in running_jobs if j["gpu_id"] == gpu_id)
        
        while gpu_running_count < 4 and jobs:
            # Check the next job
            next_job = jobs[0]
            req_mem = 8500 if next_job["img_size"] == 224 else 2500
            
            if free_mem[gpu_id] >= req_mem:
                # We can run it!
                job = jobs.pop(0)
                
                # Construct command
                log_file = f"logs_combinations/log_cifar10_{job['scale']}_k{job['k']}_r{job['img_size']}_augment.txt"
                cmd = [
                    python_bin, "run_experiment.py",
                    "--dataset", "cifar10",
                    "--n_epochs", "200",
                    "--n_runs", "1",
                    "--scale", job["scale"],
                    "--n_quantum_heads", str(job["k"]),
                    "--img_size", str(job["img_size"]),
                    "--scheduler", "onecycle",
                    "--cutmix_alpha", "1.0",
                    "--mixup_alpha", "0.8",
                    "--use_randaugment",
                    "--output_dir", "./results_combinations",
                    "--device", f"cuda:{gpu_id}",
                    "--resume"
                ]
                
                print(f"[LAUNCH] Starting Scale:{job['scale']} Size:{job['img_size']} K:{job['k']} on cuda:{gpu_id} (Free VRAM: {free_mem[gpu_id]}MiB)")
                
                # Start process
                with open(log_file, "w") as f:
                    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                
                running_jobs.append({
                    "proc": proc,
                    "gpu_id": gpu_id,
                    "scale": job["scale"],
                    "k": job["k"],
                    "img_size": job["img_size"]
                })
                
                # Update tracking vars
                gpu_running_count += 1
                free_mem[gpu_id] -= req_mem  # subtract estimated memory from local copy
            else:
                # GPU doesn't have enough memory for the next job. Stop trying to assign to this GPU.
                break

    # Update console status every 30 seconds
    current_time = time.time()
    if current_time - last_status_time >= 30:
        gpu_0_jobs = [f"{j['scale']}_k{j['k']}_r{j['img_size']}" for j in running_jobs if j["gpu_id"] == 0]
        gpu_1_jobs = [f"{j['scale']}_k{j['k']}_r{j['img_size']}" for j in running_jobs if j["gpu_id"] == 1]
        free_mem_live = get_free_memory()
        print(f"Status: GPU 0 running {len(gpu_0_jobs)}/4 {gpu_0_jobs} (Free VRAM: {free_mem_live[0]}MiB) | "
              f"GPU 1 running {len(gpu_1_jobs)}/4 {gpu_1_jobs} (Free VRAM: {free_mem_live[1]}MiB) | "
              f"Queue length: {len(jobs)}")
        last_status_time = current_time
        
    time.sleep(5)

print("All sweep jobs have been completed.")
