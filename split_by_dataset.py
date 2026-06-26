import os
import pandas as pd

# Create output directory
out_dir = "results_per_dataset"
os.makedirs(out_dir, exist_ok=True)

# Read the organized configs
df = pd.read_csv("results_organized/all_configs.csv")

# Split by dataset and save
for dataset, group in df.groupby("dataset"):
    # Sort for readability if needed
    group = group.sort_values(by="weighted_f1_mean", ascending=False)
    out_path = os.path.join(out_dir, f"{dataset}_configs.csv")
    group.to_csv(out_path, index=False)
    print(f"Saved {dataset}: {len(group)} configs to {out_path}")
