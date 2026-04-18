import json



def convert_ranking(source_file, output_file):
    json_file = source_file

    # Load JSON
    with open(json_file) as f:
        data = json.load(f)

    assignment = data["assignment"]

    # Use per-machine worker counts from JSON metadata (produced by pregglenator).
    # Compute rank offsets as prefix sums so non-uniform machines work correctly.
    # e.g. workers_per_machine=[4,4,3,4] → offsets=[0,4,8,11], total ranks=15
    if "workers_per_machine" in data:
        wpm_list = data["workers_per_machine"]
    else:
        # fallback for older JSON without the metadata field
        wpm_global = max(v["worker"] for v in assignment.values()) + 1
        num_machines = max(v["machine"] for v in assignment.values()) + 1
        wpm_list = [wpm_global] * num_machines

    rank_offset = [0] * len(wpm_list)
    for m in range(1, len(wpm_list)):
        rank_offset[m] = rank_offset[m - 1] + wpm_list[m - 1]
    total_ranks = rank_offset[-1] + wpm_list[-1]

    # Build node -> global rank mapping
    node_to_rank = {}
    for node, info in assignment.items():
        rank = rank_offset[info["machine"]] + info["worker"]
        node_to_rank[int(node)] = rank

    print(f"  workers_per_machine : {wpm_list}")
    print(f"  rank offsets        : {rank_offset}")
    print(f"  total ranks         : {total_ranks}")

    # Write partition.txt
    with open(output_file, "w") as f:
        for node in sorted(node_to_rank.keys()):
            f.write(f"{node} {node_to_rank[node]}\n")

    print(f"Generated {output_file} for {len(node_to_rank)} nodes.")
