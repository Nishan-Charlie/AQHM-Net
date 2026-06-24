import subprocess
import time
import os
from datetime import datetime

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

gpu0_jobs = [['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'micro', '--n_quantum_heads', '4', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume', '--attention_encoding', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'micro', '--n_quantum_heads', '8', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume', '--attention_encoding', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'micro', '--n_quantum_heads', '1', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume', '--ablation', 'no_quantum'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'pathmnist', '--n_epochs', '200', '--n_runs', '1', '--scale', 'micro', '--n_quantum_heads', '4', '--attention_encoding', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'small', '--n_quantum_heads', '4', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume', '--attention_encoding', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'small', '--n_quantum_heads', '8', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume', '--attention_encoding', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'medium', '--n_quantum_heads', '1', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'medium', '--n_quantum_heads', '4', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'medium', '--n_quantum_heads', '8', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:0', '--resume']]
gpu1_jobs = [['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'micro', '--n_quantum_heads', '4', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:1', '--resume', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'micro', '--n_quantum_heads', '8', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:1', '--resume', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'small', '--n_quantum_heads', '1', '--img_size', '28', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:1', '--resume', '--ablation', 'no_quantum'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'small', '--n_quantum_heads', '8', '--img_size', '28', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:1', '--resume', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'small', '--n_quantum_heads', '1', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:1', '--resume', '--ablation', 'no_quantum'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'small', '--n_quantum_heads', '4', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:1', '--resume', '--contrastive'], ['/home/e21283/miniconda3/envs/quantum/bin/python', 'run_experiment.py', '--dataset', 'cifar10', '--n_epochs', '200', '--n_runs', '1', '--scale', 'small', '--n_quantum_heads', '8', '--img_size', '224', '--scheduler', 'onecycle', '--cutmix_alpha', '1.0', '--mixup_alpha', '0.8', '--use_randaugment', '--output_dir', './results_combinations', '--device', 'cuda:1', '--resume', '--contrastive']]

def run_queue(jobs, gpu_name):
    print(f"[{ts()}] Starting queue on {gpu_name} with {len(jobs)} jobs")
    for i, cmd in enumerate(jobs):
        print(f"[{ts()}] {gpu_name} Job {i+1}/{len(jobs)}: {' '.join(cmd)}")
        
        # We will pipe output to a dynamically created log file based on args
        # Actually it's easier to just let run_experiment.py write to its default log
        # by simply executing it! Since run_experiment.py uses log_file parameter internally or stdout.
        # Wait, run_experiment.py prints to stdout, and earlier wrapper scripts piped to logs_combinations/*.txt
        
        # Let's reconstruct a log filename from args
        scale = "unknown"
        if "--scale" in cmd:
            scale = cmd[cmd.index("--scale")+1]
            
        dataset = "unknown"
        if "--dataset" in cmd:
            dataset = cmd[cmd.index("--dataset")+1]
            
        k = "unknown"
        if "--n_quantum_heads" in cmd:
            k = cmd[cmd.index("--n_quantum_heads")+1]
            
        log_name = f"logs_combinations/log_{dataset}_{scale}_k{k}_resumed_queued.txt"
        
        with open(log_name, "a") as f:
            f.write(f"\n\n================ RESUMING QUEUED JOB ON {ts()} ================\n\n")
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            proc.wait()
            
    print(f"[{ts()}] Queue on {gpu_name} FINISHED.")

if __name__ == "__main__":
    import threading
    t0 = threading.Thread(target=run_queue, args=(gpu0_jobs, "GPU 0"))
    t1 = threading.Thread(target=run_queue, args=(gpu1_jobs, "GPU 1"))
    
    t0.start()
    t1.start()
    
    t0.join()
    t1.join()
    print("ALL SEQUENTIAL QUEUES COMPLETED.")
