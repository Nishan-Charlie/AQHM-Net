# Organised Results

Deduplicated from `results_summary.csv` (50 raw runs across 2 groups -> 50 unique configs). Duplicate configs collapsed to the best instance (highest weighted-F1, then most seeds). Degenerate/collapsed runs (weighted-F1 < 0.20) are flagged and excluded from the best-per-dataset table.

## Best configuration per dataset

| Domain | Dataset | Best config | Seeds | Acc (%) | W-F1 (%) | M-F1 (%) | AUC (%) |
|---|---|---|---|---|---|---|---|
| Digits | MNIST (10-class) | K4, medium | 1 | 99.45 | 99.45 | 99.44 | 99.99 |
| Medical | BreastMNIST | no-quantum, micro, onecycle, cutmix1.0+mixup0.8 | 1 | 85.26 | 85.51 | 81.92 | 89.75 |
| Medical | DermaMNIST | no-quantum, micro | 1 | 57.41 | 63.98 | 40.48 | 88.30 |
| Medical | PathMNIST | K4+AE, micro | 1 | 90.28 | 90.00 | 87.86 | 98.85 |
| Medical | PneumoniaMNIST | K4+AE, micro, onecycle, cutmix1.0+mixup0.8 | 1 | 93.91 | 93.81 | 93.32 | 97.60 |
| Natural | CIFAR-10 | K1, micro, r224, onecycle, cutmix1.0+mixup0.8 | 1 | 88.02 | 87.97 | 87.97 | 99.07 |
| Natural | CIFAR-100 | K4 | 1 | 52.04 | 50.85 | 50.85 | 96.79 |

## Degenerate runs excluded (training collapse)

| Dataset | Config | W-F1 (%) | Source |
|---|---|---|---|
| DermaMNIST | K4, micro, onecycle, cutmix1.0+mixup0.8 | 13.52 | `results_combinations/dermamnist_k4_micro_onecycle_cutmix1.0_mixup0.8_ra2_9/summary.json` |
| DermaMNIST | no-quantum, micro, onecycle, cutmix1.0+mixup0.8 | 13.33 | `results_combinations/dermamnist_no_quantum_micro_onecycle_cutmix1.0_mixup0.8_ra2_9/summary.json` |
| DermaMNIST | K4+AE, micro, onecycle, cutmix1.0+mixup0.8 | 13.18 | `results_combinations/dermamnist_k4_attenc_micro_onecycle_cutmix1.0_mixup0.8_ra2_9/summary.json` |

## All unique configurations

Full deduplicated table in `all_configs.csv` (one row per dataset x config; `kept_source` gives the winning experiment folder, `collapsed_groups` lists the duplicates it superseded).

| Dataset | Unique configs |
|---|---|
| MNIST (10-class) | 2 |
| BreastMNIST | 9 |
| DermaMNIST | 10 |
| PathMNIST | 8 |
| PneumoniaMNIST | 7 |
| CIFAR-10 | 10 |
| CIFAR-100 | 4 |

## Experiment-group inventory (raw folders, not moved)

| Group folder | Runs (summary.json) |
|---|---|
| `results_combinations/` | 17 |
| `results_old_backbone/` | 33 |

Nothing above was moved or deleted; the raw folders remain the source of truth and `kept_source` in `all_configs.csv` points into them.

