#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# PageRank files
pr_random = "query_time_pulls_pagerank_15m_4w_FINAL_RESULTS/20260414_053820_query_times_random_15m_4w.csv"
pr_based = "query_time_pulls_pagerank_15m_4w_FINAL_RESULTS/20260414_075531_query_times_BASED_15w_4m_pagerank.csv"
pr_compute = "query_time_pulls_pagerank_15m_4w_FINAL_RESULTS/20260414_055306_query_times_compute_only_pagerank_15m_4w.csv"

# SSSP files
sssp_random = "query_times_pulls_sssp_15m_4w_FINAL_RESULTS/query_times_20260403_112926_random_15m_4w.csv"
sssp_based = "query_times_pulls_sssp_15m_4w_FINAL_RESULTS/query_times_20260414_080939_BASED_15w_4m_sssp_12src.csv"
sssp_compute = "query_times_pulls_sssp_15m_4w_FINAL_RESULTS/query_times_20260403_112926_sssp_compute_only_15m_4w.csv"

FILES = {
    "PageRank": {
        "Random": pr_random,
        "CRISP": pr_compute,
        "BASED": pr_based,
    },
    "SSSP": {
        "Random": sssp_random,
        "CRISP": sssp_compute,
        "BASED": sssp_based,
    },
}

DISPLAY_ORDER = ["Random", "CRISP", "BASED"]

# Colors
BOX_COLORS = {
    "Random": "tab:blue",
    "CRISP": "tab:orange",
    "BASED": "tab:green",
}


def load_query_times(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, on_bad_lines="skip")

    return pd.to_numeric(df["query_seconds"], errors="coerce").dropna().astype(float).values


def annotate_stats(ax, values):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for i, vals in enumerate(values, start=1):
        median = np.median(vals)
        mean = np.mean(vals)
        p99 = np.percentile(vals, 99)

        text = (
            f"med={median:.2f}\n"
            f"mean={mean:.2f}\n"
            f"p99={p99:.2f}"
        )

        ax.annotate(
            text,
            xy=(i, p99),
            xytext=(i, p99 + 0.04 * y_range),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

def plot_boxplots(all_data, output_path="query_times_clean.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, workload in zip(axes, ["PageRank", "SSSP"]):
        data = all_data[workload]

        labels = [l for l in DISPLAY_ORDER if l in data]
        values = [data[l] for l in labels]

        bp = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
            showmeans=True,
            medianprops=dict(color="black", linewidth=2),
            meanprops=dict(
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=5,
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor="none",
                markeredgecolor="gray",
                markersize=3,
                alpha=0.4,
            ),
        )

        # Color boxes
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(BOX_COLORS[label])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Query Time (seconds)")
        ax.set_title(f"{workload} Query Time Distribution")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Padding so annotations fit nicely
        y_low = min(np.min(v) for v in values)
        y_high = max(np.max(v) for v in values)
        pad = (y_high - y_low) * 0.15
        ax.set_ylim(y_low - pad, y_high + pad)

        annotate_stats(ax, values)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")
    plt.show()

def main():
    all_data = {}

    for workload, files in FILES.items():
        all_data[workload] = {}
        for label, path in files.items():
            all_data[workload][label] = load_query_times(path)

    plot_boxplots(all_data)


if __name__ == "__main__":
    main()