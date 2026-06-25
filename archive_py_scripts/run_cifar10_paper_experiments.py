import subprocess
import time
import sys
import os

# Create log directory
os.makedirs("logs_combinations", exist_ok=True)

# Path to python environment
python_bin = "/home/e21283/miniconda3/envs/quantum/bin/python"
output_dir = "./results_combinations"

# Define only the important experiments for the Q1 paper:
# 1. Full Proposed AQHM-Net (with attention_encoding and contrastive)
# 2. No-Quantum Ablation
# 3. No-Attention Encoding Ablation (Arctan encoding baseline)
# 4. No-Contrastive Ablation (for RGB)
jobs = []

# Helper to add a job
def add_job(scale, img_size, k, ablation=None, attenc=False, contrastive=False):
    jobs.append({
        "scale": scale,
        "img_size": img_size,
        "k": k,
        "ablation": ablation,
        "attention_encoding": attenc,
        "contrastive": contrastive
    })

# Define scales and resolutions of interest
scales = ["micro", "small"]
img_sizes = [28, 224]

for scale in scales:
    for img_size in img_sizes:
        # A. Full Proposed Model (K=4, K=8)
        for k in [4, 8]:
            add_job(scale, img_size, k, attenc=True, contrastive=True)
            
        # B. No-Quantum Baseline
        add_job(scale, img_size, k=1, ablation="no_quantum", attenc=False, contrastive=False)
        
        # C. No-Attention Encoding Ablation (standard Arctan encoding, K=4, K=8)
        for k in [4, 8]:
            add_job(scale, img_size, k, attenc=False, contrastive=True)

    # D. No-Contrastive Ablation (at K=4, size 28)
    add_job(scale, 28, k=4, attenc=True, contrastive=False)


def get_vram_requirement(job):
    scale = job["scale"]
    img_size = job["img_size"]
    k = job["k"]
    
    if img_size == 224:
        base = 9000
    else:
        if scale == "micro":
            base = 1800
        elif scale == "small":
            base = 2500
        elif scale == "medium":
            base = 4000
        else: # large
            base = 6500
    return base + 200 * k

def get_free_memory():
    try:
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        res = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        return [int(x) for x in res]
    except Exception as e:
        print(f"Error querying GPU memory: {e}")
        return [0, 0]

print(f"Loaded {len(jobs)} Q1 Paper Sweep jobs in the queue.")
print("Script is prepared and ready to run, but not executed.")

# The execution scheduler logic is preserved below so that you can simply run this script.
def run_scheduler():
    running_jobs = []
    last_status_time = 0
    
    # Copy queue to execute
    queue = list(jobs)
    
    while queue or running_jobs:
        # 1. Check on running jobs
        still_running = []
        for j in running_jobs:
            ret = j["proc"].poll()
            if ret is not None:
                if ret == 0:
                    print(f"[SUCCESS] Job {j['desc']} completed.")
                else:
                    print(f"[FAILED] Job {j['desc']} exited with code {ret}.")
            else:
                still_running.append(j)
        running_jobs = still_running

        # 2. Assign new jobs to GPUs if VRAM is available
        free_mem = get_free_memory()
        
        for gpu_id in [0, 1]:
            gpu_running_count = sum(1 for j in running_jobs if j["gpu_id"] == gpu_id)
            
            while gpu_running_count < 8 and queue:
                next_job = queue[0]
                req_mem = get_vram_requirement(next_job)
                
                if free_mem[gpu_id] >= req_mem:
                    job = queue.pop(0)
                    
                    # Tag building for logs and commands
                    desc = f"{job['scale']}_k{job['k']}_r{job['img_size']}"
                    if job["ablation"]:
                        desc += f"_{job['ablation']}"
                    if job["attention_encoding"]:
                        desc += "_attenc"
                    if not job["contrastive"] and job["k"] > 1 and job["ablation"] is None:
                        desc += "_nocontrastive"
                        
                    log_file = f"logs_combinations/log_cifar10_{desc}_paper.txt"
                    
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
                        "--output_dir", output_dir,
                        "--device", f"cuda:{gpu_id}",
                        "--resume"
                    ]
                    
                    if job["ablation"]:
                        cmd += ["--ablation", job["ablation"]]
                    if job["attention_encoding"]:
                        cmd += ["--attention_encoding"]
                    if job["contrastive"]:
                        cmd += ["--contrastive"]
                        
                    print(f"[LAUNCH] Starting {desc} on cuda:{gpu_id} (Free VRAM: {free_mem[gpu_id]}MiB)")
                    
                    with open(log_file, "w") as f:
                        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                    
                    running_jobs.append({
                        "proc": proc,
                        "gpu_id": gpu_id,
                        "desc": desc,
                        "req_mem": req_mem
                    })
                    
                    gpu_running_count += 1
                    free_mem[gpu_id] -= req_mem
                else:
                    break

        current_time = time.time()
        if current_time - last_status_time >= 30:
            gpu_0_jobs = [j["desc"] for j in running_jobs if j["gpu_id"] == 0]
            gpu_1_jobs = [j["desc"] for j in running_jobs if j["gpu_id"] == 1]
            free_mem_live = get_free_memory()
            print(f"Status: GPU 0 running {len(gpu_0_jobs)}/8 {gpu_0_jobs} (Free VRAM: {free_mem_live[0]}MiB) | "
                  f"GPU 1 running {len(gpu_1_jobs)}/8 {gpu_1_jobs} (Free VRAM: {free_mem_live[1]}MiB) | "
                  f"Queue length: {len(queue)}")
            last_status_time = current_time
            
        time.sleep(5)
    print("All sweep jobs completed.")

if __name__ == "__main__":
    run_scheduler()
