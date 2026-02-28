import json

# Path to pregglenator output
json_file = "_large_twitch_assignment_ours.json"

# Load JSON
with open(json_file) as f:
    data = json.load(f)

assignment = data["assignment"]

# Determine workers per machine
workers_per_machine = max(v["worker"] for v in assignment.values()) + 1

# Convert machine/worker -> global rank
def machine_worker_to_rank(machine, worker, workers_per_machine):
    return machine * workers_per_machine + worker

# Build node -> global rank mapping
node_to_rank = {}
for node, info in assignment.items():
    machine = info["machine"]
    worker = info["worker"]
    rank = machine_worker_to_rank(machine, worker, workers_per_machine)
    node_to_rank[int(node)] = rank

# Write partition.txt
with open("partition.txt", "w") as f:
    for node in sorted(node_to_rank.keys()):
        f.write(f"{node} {node_to_rank[node]}\n")

print(f"Generated partition.txt for {len(node_to_rank)} nodes.")
