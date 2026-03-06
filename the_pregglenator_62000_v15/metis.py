"""
metis.py - machine partitioning via METIS with uniform weights, random worker assignment

All edge weights are set to 1 before partitioning.
METIS is run only at the machine level.
Workers are randomly assigned within each machine.
"""

import argparse
import json
import math
import random
from collections import defaultdict
import pymetis


def load_comm_json(path):
    with open(path, "r") as f:
        raw = json.load(f)
    comm = {}
    for s, nbrs in raw.items():
        s_i = int(s)
        comm[s_i] = {}
        for t, w in nbrs.items():
            comm[s_i][int(t)] = float(w)
    return comm


def load_comm_edgelist(path, directed=True):
    comm = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            s = int(parts[0])
            t = int(parts[1])
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            comm[s][t] = comm[s].get(t, 0.0) + w
            if not directed:
                comm[t][s] = comm[t].get(s, 0.0) + w
    return dict(comm)


def infer_num_nodes(comm, explicit_n=None):
    if explicit_n is not None:
        return explicit_n
    mx = -1
    for s, nbrs in comm.items():
        mx = max(mx, s)
        for t in nbrs.keys():
            mx = max(mx, t)
    return mx + 1


def symmetrize_unweighted(comm, n):
    """
    Build undirected adjacency with all edge weights set to 1.
    """
    adj = [set() for _ in range(n)]
    for u, nbrs in comm.items():
        if u < 0 or u >= n:
            continue
        for v in nbrs.keys():
            if v < 0 or v >= n or v == u:
                continue
            adj[u].add(v)
            adj[v].add(u)

    und = [defaultdict(float) for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            und[u][v] = 1.0
    return und


def build_pymetis_inputs(und_adj):
    n = len(und_adj)
    xadj = [0]
    adjncy = []
    eweights = []
    for u in range(n):
        items = sorted(und_adj[u].items(), key=lambda kv: kv[0])
        for v, w in items:
            adjncy.append(v)
            eweights.append(int(round(w)))
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights


def metis_partition(und_adj, nparts):
    xadj, adjncy, eweights = build_pymetis_inputs(und_adj)
    _, parts = pymetis.part_graph(
        nparts,
        xadj=xadj,
        adjncy=adjncy,
        eweights=eweights,
    )
    return list(parts)


def compute_cut_weight(und_adj, part_of):
    cut = 0.0
    seen = set()
    for u in range(len(und_adj)):
        for v, w in und_adj[u].items():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            if part_of[u] != part_of[v]:
                cut += w
    return cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--format", choices=["json", "edgelist"], required=True)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--num_nodes", type=int, default=None)
    ap.add_argument("--num_machines", type=int, required=True)
    ap.add_argument("--num_workers", type=int, default=4, help="Workers per machine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    random.seed(args.seed)

    if args.format == "json":
        comm = load_comm_json(args.input)
    else:
        comm = load_comm_edgelist(args.input, directed=args.directed)

    n = infer_num_nodes(comm, explicit_n=args.num_nodes)
    print(f"Nodes: {n}, Machines: {args.num_machines}, Workers/machine: {args.num_workers}")

    # All weights set to 1
    und_adj = symmetrize_unweighted(comm, n)

    # METIS partitioning at machine level only
    machine_of = metis_partition(und_adj, args.num_machines)

    # Collect nodes per machine
    nodes_in_machine = [[] for _ in range(args.num_machines)]
    for u, m in enumerate(machine_of):
        nodes_in_machine[m].append(u)

    # Random worker assignment within each machine
    worker_of = [-1] * n
    for m in range(args.num_machines):
        nodes_m = nodes_in_machine[m]
        shuffled = nodes_m[:]
        random.shuffle(shuffled)
        for i, u in enumerate(shuffled):
            worker_of[u] = i % args.num_workers

    # Build output
    out = {}
    for u in range(n):
        out[str(u)] = {"machine": int(machine_of[u]), "worker": int(worker_of[u])}

    machine_sizes = [len(nodes_in_machine[m]) for m in range(args.num_machines)]
    stats = {
        "num_nodes": n,
        "num_machines": args.num_machines,
        "num_workers_per_machine": args.num_workers,
        "machine_cut_weight": compute_cut_weight(und_adj, machine_of),
        "machine_sizes": machine_sizes,
    }

    payload = {"assignment": out, "stats": stats}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Machine cut weight (unweighted): {stats['machine_cut_weight']}")
    print(f"Machine sizes: {machine_sizes}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
