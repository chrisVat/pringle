import glob
import csv
import json
from collections import defaultdict

# nested dict: comm[src][dst] = total_count
comm = defaultdict(lambda: defaultdict(float))

# read merged file
input_file = "all_merged_agg.csv"

with open(input_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = int(row["src_vertex"])
        dst = int(row["dst_vertex"])
        count = float(row["count"])
        comm[src][dst] += count

# convert to normal dict of dicts (JSON-serializable)
output = {str(src): {str(dst): weight
                     for dst, weight in neighbors.items()}
          for src, neighbors in comm.items()}

with open("vertex_comm.json", "w") as f:
    json.dump(output, f, indent=2)

print("Wrote vertex_comm.json")
