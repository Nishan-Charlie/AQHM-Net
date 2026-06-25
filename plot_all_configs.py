import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Publication style
_FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family":           _FONT,
    "font.size":             9,
    "axes.titlesize":        9,
    "axes.titleweight":      "bold",
    "axes.labelsize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
})

_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0C030", "#555555"]

DATASET_PRETTY = {
    "mnist": "MNIST-10", "mnist_0123": "MNIST-4",
    "pathmnist": "PathMNIST", "dermamnist": "DermaMNIST",
    "pneumoniamnist": "PneumoniaMNIST", "breastmnist": "BreastMNIST",
    "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
}

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_organized"
PLOTS_DIR = RESULTS_DIR / "plots"

def load_data():
    all_configs = pd.read_csv(RESULTS_DIR / "all_configs.csv")
    all_configs['dataset_pretty'] = all_configs['dataset'].map(lambda x: DATASET_PRETTY.get(x, x))
    return all_configs

def save_plot(fig, filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_best_per_dataset(df):
    best_df = df[~df['degenerate']].loc[df[~df['degenerate']].groupby('dataset')['weighted_f1_mean'].idxmax()]
    
    # Sort
    domain_order = {"Digits": 0, "Medical": 1, "Natural": 2}
    best_df['domain_order'] = best_df['domain'].map(domain_order)
    best_df = best_df.sort_values(['domain_order', 'dataset'])
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(len(best_df))
    w = 0.35
    
    ax.bar([i - w/2 for i in x], best_df['weighted_f1_mean'], w, 
           color=_PALETTE[0], label='Weighted F1', edgecolor="white", zorder=3)
    ax.bar([i + w/2 for i in x], best_df['weighted_accuracy_mean'], w, 
           color=_PALETTE[1], label='Weighted Accuracy', edgecolor="white", zorder=3)
    
    ax.grid(axis='y', ls='--', alpha=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(best_df['dataset_pretty'], rotation=30, ha='right')
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.set_title("Best Configurations per Dataset")
    
    save_plot(fig, "best_per_dataset.png")

def plot_scale_effect(df):
    # Filter datasets that have varying scale
    scale_df = df[df['dataset'] == 'cifar10'].copy()
    if scale_df.empty: return
    
    # Filter for standard K=1 or K=4
    scale_df = scale_df[scale_df['K'].isin([1, 4])]
    scale_df['Scale'] = pd.Categorical(scale_df['scale'], categories=['micro', 'small', 'medium', 'large'], ordered=True)
    scale_df = scale_df.dropna(subset=['Scale', 'weighted_accuracy_mean']).sort_values(['K', 'Scale'])
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.lineplot(data=scale_df, x='Scale', y='weighted_accuracy_mean', hue='K', marker='o', ax=ax, palette=[_PALETTE[0], _PALETTE[2]])
    ax.set_ylabel("Weighted Accuracy")
    ax.set_title("Effect of Backbone Scale (CIFAR-10)")
    ax.grid(True, ls='--', alpha=0.5)
    save_plot(fig, "effect_of_scale.png")

def plot_k_effect(df):
    k_df = df[df['K'].isin([1, 2, 4, 8])].copy()
    # Average across small models to see general trend
    k_df = k_df[(k_df['scale'] == 'small') & (~k_df['attention_encoding'])]
    
    if k_df.empty: return
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(data=k_df, x='dataset_pretty', y='weighted_accuracy_mean', hue='K', ax=ax, palette="Blues_d")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_ylabel("Weighted Accuracy")
    ax.set_xlabel("")
    ax.legend(title='Parallel Circuits (K)')
    ax.set_title("Effect of Parallel Circuits (small backbone)")
    ax.grid(axis='y', ls='--', alpha=0.5)
    save_plot(fig, "effect_of_k.png")

def main():
    if not (RESULTS_DIR / "all_configs.csv").exists():
        print("all_configs.csv not found. Please run organize_results.py first.")
        return
    
    df = load_data()
    plot_best_per_dataset(df)
    plot_scale_effect(df)
    plot_k_effect(df)
    
if __name__ == "__main__":
    main()
