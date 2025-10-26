# ===============================================================
# Visualization Script for Quantum Kernel Experiment Results
# ===============================================================
# This script:
# 1. Loads summary results
# 2. Generates multiple plots (bar, gain, heatmap, radar)
# 3. Saves all figures in a 'plots/' directory with clear names
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------
# Setup: ensure plots directory exists
# ----------------------------------------------------------------
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
print(f"Plots will be saved in: {PLOTS_DIR.resolve()}\n")


# ----------------------------------------------------------------
# Load summary results
# ----------------------------------------------------------------
def load_summary(filepath="summary_results.csv"):
    """Load summary results into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    print("\nLoaded summary results:\n", df.head())
    return df


# ----------------------------------------------------------------
# Plot baseline vs optimized comparison
# ----------------------------------------------------------------
def plot_baseline_vs_final(df):
    """Compare baseline and optimized accuracies for all datasets."""
    df_melted = df.melt(
        id_vars="Dataset",
        value_vars=["Baseline", "Final"],
        var_name="Metric",
        value_name="Accuracy"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Dataset", y="Accuracy", hue="Metric", data=df_melted, palette="viridis")
    plt.title("Baseline vs Optimized Accuracy Across Datasets")
    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Save figure
    save_path = PLOTS_DIR / "baseline_vs_optimized_accuracy.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()


# ----------------------------------------------------------------
# Plot percentage gain
# ----------------------------------------------------------------
def plot_gain(df):
    """Plot the percentage improvement (Gain %) for each dataset."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Dataset", y="Gain (%)", hue="Dataset", data=df, palette="crest", legend=False)
    plt.title("Performance Gain After Optimization")
    plt.ylabel("Gain (%)")
    plt.xlabel("Dataset")
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Save figure
    save_path = PLOTS_DIR / "performance_gain.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()


# ----------------------------------------------------------------
# Plot normalized heatmap
# ----------------------------------------------------------------
def plot_summary_heatmap(df):
    """Plot a normalized heatmap showing relative performance."""
    df_norm = df.copy()
    df_norm["Normalized Accuracy"] = df_norm["Final"] / df_norm["Baseline"]

    df_norm_melted = df_norm.melt(
        id_vars=["Dataset"],
        value_vars=["Baseline", "Final", "Normalized Accuracy"],
        var_name="Metric",
        value_name="Value"
    )

    # Correct pivot syntax
    pivot_df = df_norm_melted.pivot(index="Dataset", columns="Metric", values="Value")

    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title("Normalized Accuracy Heatmap")
    plt.ylabel("Dataset")
    plt.xlabel("Metric")
    plt.tight_layout()

    # Save figure
    save_path = PLOTS_DIR / "normalized_accuracy_heatmap.png"
    plt.savefig(save_path, dpi=300)
    print(f"🔥 Saved: {save_path}")
    plt.show()


# ----------------------------------------------------------------
# Plot radar chart
# ----------------------------------------------------------------
def plot_radar_chart(df):
    """Plot a radar chart comparing datasets on normalized performance."""
    categories = list(df["Dataset"])
    values = list(df["Final"] / df["Baseline"])

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="teal", linewidth=2)
    ax.fill(angles, values, color="teal", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    plt.title("Normalized Accuracy Radar Plot", size=14)
    plt.tight_layout()

    # Save figure
    save_path = PLOTS_DIR / "normalized_accuracy_radar.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()


# ----------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------
if __name__ == "__main__":
    summary_path = Path("summary_results.csv")

    if not summary_path.exists():
        # Create a sample summary if not found
        data = {
            "Dataset": ["PROTEINS", "MUTAG", "AIDS", "NCI1", "PTC_MR"],
            "Baseline": [0.7304, 0.8617, 0.9980, 0.7401, 0.5785],
            "Final": [0.7502, 0.8781, 0.9980, 0.7416, 0.5994],
        }
        df = pd.DataFrame(data)
        df["Gain (%)"] = ((df["Final"] - df["Baseline"]) / df["Baseline"]) * 100
        df.to_csv(summary_path, index=False)
        print("Sample summary_results.csv created.")

    df = load_summary(summary_path)

    plot_baseline_vs_final(df)
    plot_gain(df)
    plot_summary_heatmap(df)
    plot_radar_chart(df)
