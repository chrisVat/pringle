#!/usr/bin/env python3
import subprocess
import csv
import io
from collections import defaultdict

# Get list of source directories from HDFS
result = subprocess.run(
    ["hdfs", "dfs", "-ls", "/comm_traces"],
    capture_output=True, text=True
)

src_dirs = [
    line.split()[-1]
    for line in result.stdout.strip().split("\n")
    if "src_" in line
]

print(f"Found {len(src_dirs)} source directories")

# Aggregate comm[src][dst] = total count across all source runs
comm = defaultdict(lambda: defaultdict(int))

for src_dir in src_dirs:
    print(f"Processing {src_dir}...")
    result = subprocess.run(
        ["hdfs", "dfs", "-cat", f"{src_dir}/merged.csv"],
        capture_output=True, text=True
    )
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        src = int(row["src_vertex"])
        dst = int(row["dst_vertex"])
        count = int(row["count"])
        comm[src][dst] += count

# Write aggregated output
output_path = "/home/ubuntu/all_merged_agg.csv"
total_edges = 0
with open(output_path, "w") as f:
    f.write("src_vertex,dst_vertex,count\n")
    for src, neighbors in comm.items():
        for dst, weight in neighbors.items():
            f.write(f"{src},{dst},{weight}\n")
            total_edges += 1

print(f"Done! {total_edges} unique (src,dst) pairs")
print(f"Saved to {output_path}")