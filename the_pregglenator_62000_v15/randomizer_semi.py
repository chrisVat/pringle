import argparse
import json
from collections import defaultdict


def generate_assignment(num_nodes: int, num_machines: int, num_workers: int) -> dict:
    assignment = {}
    cur_machine = 0
    for i in range(num_nodes):
        assignment[str(i)] = {
            "machine": cur_machine,
            "worker":  i % num_workers,
        }
        # if i is perfectly divisible by num_workers * num_machines, move to next machine
        if (i + 1) % (num_workers * num_machines) == 0:
            cur_machine += 1
            if cur_machine >= num_machines:
                cur_machine = 0
    return {"assignment": assignment}


def main():
    parser = argparse.ArgumentParser(
        description="Generate a cyclic node-to-worker/machine assignment JSON."
    )
    parser.add_argument("--num_nodes",    type=int, default=168114, help="Total number of nodes")
    parser.add_argument("--num_machines", type=int, default=4,      help="Number of machines")
    parser.add_argument("--num_workers",  type=int, default=4,      help="Number of workers per machine")
    parser.add_argument("--output",       type=str, default="semi_random.json", help="Output file path")
    args = parser.parse_args()

    if args.num_nodes < 1:
        parser.error("num_nodes must be >= 1")
    if args.num_machines < 1:
        parser.error("num_machines must be >= 1")
    if args.num_workers < 1:
        parser.error("num_workers must be >= 1")

    data = generate_assignment(args.num_nodes, args.num_machines, args.num_workers)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

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