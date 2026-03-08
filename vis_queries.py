import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_DIR = "QUERY_RESULTS/"

NAME_MAP = {
    "semi_random": "semi_random",
    "greedmax": "greedy",
    "compute_only": "compute_min",
    "random": "random",
    "metis": "metis",
    "gamer_mode_strict": "metis_gaming",
    "pregglenator": "double_weighted_metis",
}

def get_label(filename):
    base = os.path.basename(filename)
    # Strip prefix "query_times_" and extension
    suffix = base.removeprefix("query_times_").removesuffix(".csv")
    for key, label in NAME_MAP.items():
        if suffix == key or suffix == f"pregglenator_{key}":
            return label
    return suffix

def load_data():
    files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    data = {}
    for f in files:
        label = get_label(f)
        df = pd.read_csv(f, on_bad_lines="skip")
        # Drop rows where query_seconds is not a valid number (footer lines)
        df = df[pd.to_numeric(df["query_seconds"], errors="coerce").notna()]
        data[label] = df["query_seconds"].astype(float).values
    return data

def main():
    data = load_data()

    # Desired display order
    order = ["random", "semi_random", "compute_min", "greedy", "metis", "metis_gaming", "double_weighted_metis"]
    labels = [l for l in order if l in data] + [l for l in data if l not in order]
    values = [data[l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(values, patch_artist=True, medianprops=dict(color="black", linewidth=2))

    colors = plt.cm.tab10.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay median, mean, min, max annotations
    for i, (vals, label) in enumerate(zip(values, labels), start=1):
        median = np.median(vals)
        mean = np.mean(vals)
        mn = np.min(vals)
        mx = np.max(vals)
        ax.annotate(f"med={median:.2f}\nmean={mean:.2f}\nmin={mn:.2f}\nmax={mx:.2f}",
                    xy=(i, mx), xytext=(i, mx + 0.05),
                    fontsize=6.5, ha="center", va="bottom", color="dimgray")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Query Time (seconds)")
    ax.set_title("Query Time Distribution by Partitioning Method")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("query_times_boxplot.png", dpi=150)
    print("Saved query_times_boxplot.png")
    plt.show()

if __name__ == "__main__":
    main()
