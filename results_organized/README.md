# Organised Results

Deduplicated from `results_summary.csv` (60 raw runs across 14 groups -> 45 unique configs). Duplicate configs collapsed to the best instance (highest weighted-F1, then most seeds). Degenerate/collapsed runs (weighted-F1 < 0.20) are flagged and excluded from the best-per-dataset table.

## Best configuration per dataset

| Domain | Dataset | Best config | Seeds | Acc (%) | W-F1 (%) | M-F1 (%) | AUC (%) |
|---|---|---|---|---|---|---|---|
| Digits | MNIST (10-class) | K4, medium | 1 | 99.45 | 99.45 | 99.44 | 99.99 |
| Digits | MNIST-4 (0-3) | K1, 300ep | 1 | 99.78 | 99.78 | 99.78 | 99.97 |
| Medical | BreastMNIST | K4+AE, micro, onecycle, cutmix1.0+mixup0.8 | 1 | 82.05 | 82.67 | 78.94 | 91.52 |
| Medical | DermaMNIST | K1, 300ep | 1 | 70.17 | 72.29 | 54.35 | 91.96 |
| Medical | PathMNIST | K1 | 10 | 81.38 ± 2.61 | 81.15 ± 2.52 | 77.94 ± 1.91 | 97.49 ± 0.35 |
| Medical | PneumoniaMNIST | K4+AE, micro, onecycle, cutmix1.0+mixup0.8 | 1 | 93.91 | 93.81 | 93.32 | 97.60 |
| Natural | CIFAR-10 | K4, large | 1 | 89.55 | 89.59 | 89.59 | 98.88 |
| Natural | CIFAR-100 | K1, 300ep | 1 | 53.30 | 52.12 | 52.12 | 96.92 |

## Degenerate runs excluded (training collapse)

| Dataset | Config | W-F1 (%) | Source |
|---|---|---|---|
| MNIST (10-class) | K1 | 1.76 | `results/mnist/summary.json` |
| MNIST (10-class) | K1, mixup0.8 | 1.76 | `results/mnist_cosine_restarts_mixup0.8/summary.json` |
| DermaMNIST | K4+AE, micro, onecycle, cutmix1.0+mixup0.8 | 13.18 | `results_combinations/dermamnist_k4_attenc_micro_onecycle_cutmix1.0_mixup0.8_ra2_9/summary.json` |

## All unique configurations

Full deduplicated table in `all_configs.csv` (one row per dataset x config; `kept_source` gives the winning experiment folder, `collapsed_groups` lists the duplicates it superseded).

| Dataset | Unique configs |
|---|---|
| MNIST (10-class) | 4 |
| MNIST-4 (0-3) | 3 |
| BreastMNIST | 4 |
| DermaMNIST | 7 |
| PathMNIST | 1 |
| PneumoniaMNIST | 4 |
| CIFAR-10 | 13 |
| CIFAR-100 | 9 |

## Experiment-group inventory (raw folders, not moved)

| Group folder | Runs (summary.json) |
|---|---|
| `results/` | 9 |
| `results_300ep/` | 5 |
| `results_attenc/` | 1 |
| `results_attenc_ab/` | 2 |
| `results_cifar100_k1/` | 2 |
| `results_cifar100_k4/` | 2 |
| `results_cifar100_k8/` | 2 |
| `results_cifar_sweep/` | 7 |
| `results_combinations/` | 11 |
| `results_kcompare/` | 5 |
| `results_parallel/` | 9 |
| `results_res/` | 1 |
| `results_res128/` | 0  *(empty -- safe to delete)* |
| `results_scale/` | 3 |
| `results_scale_k4/` | 1 |

Nothing above was moved or deleted; the raw folders remain the source of truth and `kept_source` in `all_configs.csv` points into them.

