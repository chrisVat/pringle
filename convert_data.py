"""
convert_data.py - Convert comm_traces CSV to pregglenator-ready edgelist.

Aggregates out superstep: sums count over all supersteps for each (src_vertex, dst_vertex) pair.
Output format (space-separated): src dst weight
"""

import argparse
import csv
import os
from collections import defaultdict


def convert(input_path, output_path):
    edges = defaultdict(float)

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row["src_vertex"])
            dst = int(row["dst_vertex"])
            count = float(row["count"])
            edges[(src, dst)] += count

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for (src, dst), weight in sorted(edges.items()):
            f.write(f"{src} {dst} {weight}\n")

    print(f"Written {len(edges)} edges to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="comm_traces/src_9783/merged.csv")
    parser.add_argument("output", nargs="?", default="pregglenator_ready/src_9783.edgelist")
    args = parser.parse_args()

    convert(args.input, args.output)
