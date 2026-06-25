$OUT = "./results_scale_k4"
$COMMON = "--dataset cifar10 --n_runs 1 --base_seed 0 --n_epochs 100 --patience 1000 --output_dir $OUT"

$date = Get-Date -Format "HH:mm:ss"
Write-Host ">>> START $date"

if (!(Test-Path -Path $OUT)) {
    New-Item -ItemType Directory -Force -Path $OUT | Out-Null
}

python run_experiment.py --dataset cifar10 --n_runs 1 --base_seed 0 --n_epochs 100 --patience 1000 --output_dir $OUT --scale large --n_quantum_heads 4 > "$OUT/large.log" 2>&1
# python run_experiment.py --dataset cifar10 --n_runs 1 --base_seed 0 --n_epochs 100 --patience 1000 --output_dir $OUT --scale medium > "$OUT/medium.log" 2>&1
# python run_experiment.py --dataset cifar10 --n_runs 1 --base_seed 0 --n_epochs 100 --patience 1000 --output_dir $OUT --scale small > "$OUT/small.log" 2>&1

$date = Get-Date -Format "HH:mm:ss"
Write-Host ">>> ALL DONE $date"
