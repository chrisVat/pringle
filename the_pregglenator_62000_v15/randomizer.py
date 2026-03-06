import argparse
import json
import random


def generate_assignment(num_nodes: int, num_machines: int, num_workers: int) -> dict:
    # Build a balanced list of (machine, worker) pairs across all nodes.
    # Each node gets one pair. We distribute as evenly as possible.

    num_slots = num_machines * num_workers

    # Base count of nodes per slot, and how many slots get one extra
    base, remainder = divmod(num_nodes, num_slots)

    slots = []
    for machine in range(num_machines):
        for worker in range(num_workers):
            slots.append({"machine": machine, "worker": worker})

    # Build the full pool: each slot appears base times, first `remainder` slots get +1
    pool = []
    for i, slot in enumerate(slots):
        count = base + (1 if i < remainder else 0)
        pool.extend([slot] * count)

    random.shuffle(pool)

    assignment = {}
    for node_id, slot in enumerate(pool):
        assignment[str(node_id)] = slot

    return {"assignment": assignment}


def main():
    parser = argparse.ArgumentParser(
        description="Generate a balanced node-to-worker/machine assignment JSON."
    )
    parser.add_argument("--num_nodes",    type=int, default=168114, help="Total number of nodes")
    parser.add_argument("--num_machines", type=int, default=4, help="Number of machines")
    parser.add_argument("--num_workers",  type=int, default=4, help="Number of workers per machine")
    parser.add_argument("--output",       type=str, default="random.json", help="Output file path (default: assignment.json)")
    parser.add_argument("--seed",         type=int, default=None, help="Optional random seed for reproducibility")
    args = parser.parse_args()

    if args.num_nodes < 1:
        parser.error("num_nodes must be >= 1")
    if args.num_machines < 1:
        parser.error("num_machines must be >= 1")
    if args.num_workers < 1:
        parser.error("num_workers must be >= 1")

    if args.seed is not None:
        random.seed(args.seed)

    data = generate_assignment(args.num_nodes, args.num_machines, args.num_workers)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    # Print balance summary
    from collections import defaultdict
    machine_counts = defaultdict(int)
    worker_counts  = defaultdict(int)
    for entry in data["assignment"].values():
        machine_counts[entry["machine"]] += 1
        worker_counts[entry["worker"]]   += 1

    print(f"Written {args.num_nodes} nodes to '{args.output}'")
    print(f"\nNodes per machine: { {f'm{k}': v for k, v in sorted(machine_counts.items())} }")
    print(f"Nodes per worker:  { {f'w{k}': v for k, v in sorted(worker_counts.items())} }")


if __name__ == "__main__":
    main()