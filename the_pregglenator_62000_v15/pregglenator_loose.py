"""
pregglenator 62000 v15b - two-level partitioning with pymetis
Change vs v15:
  - Stage 2 workers are NOT strict capacity.
  - We allow slight imbalance around a target nodes_per_worker via --worker_size_tol.
  - Optional fixed --workers_per_machine (exact count) OR auto workers = ceil(k/target).

Stage 1 machines remains strict capacity via repair_capacity (same as your v15).
"""

import argparse
import json
import math
from collections import defaultdict, deque
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
    """
    Edge list format:
      src dst weight
    Lines starting with # are ignored.
    """
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


def symmetrize_to_undirected(comm, n):
    """
    Returns undirected adjacency with weights:
      und[u][v] = comm[u][v] + comm[v][u], u != v
    """
    adj = [defaultdict(float) for _ in range(n)]
    for u, nbrs in comm.items():
        if u < 0 or u >= n:
            continue
        for v, w in nbrs.items():
            if v < 0 or v >= n or v == u:
                continue
            adj[u][v] += float(w)

    und = [defaultdict(float) for _ in range(n)]
    for u in range(n):
        for v, w_uv in adj[u].items():
            w_vu = adj[v].get(u, 0.0)
            w = w_uv + w_vu
            if w <= 0:
                continue
            und[u][v] = w
            und[v][u] = w
    return und


def build_pymetis_inputs(und_adj):
    """
    pymetis expects:
      xadj: offsets
      adjncy: neighbor ids
      eweights: edge weights aligned with adjncy
    """
    n = len(und_adj)
    xadj = [0]
    adjncy = []
    eweights = []
    for u in range(n):
        items = sorted(und_adj[u].items(), key=lambda kv: kv[0])
        for v, w in items:
            adjncy.append(v)
            eweights.append(int(max(1, round(w))))
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights


def metis_partition(und_adj, nparts, seed=42):
    xadj, adjncy, eweights = build_pymetis_inputs(und_adj)
    _, parts = pymetis.part_graph(
        nparts,
        xadj=xadj,
        adjncy=adjncy,
        eweights=eweights,
    )
    return list(parts)


def compute_boundary_nodes(und_adj, part_of):
    boundary = set()
    for u in range(len(und_adj)):
        pu = part_of[u]
        for v in und_adj[u].keys():
            if part_of[v] != pu:
                boundary.add(u)
                break
    return boundary


def move_cost_delta(und_adj, u, src_part, dst_part, part_of):
    """
    Cost delta for moving u from src_part to dst_part under cut objective.
    Negative means improvement.
    """
    delta = 0.0
    for v, w in und_adj[u].items():
        pv = part_of[v]
        before_cut = 1 if pv != src_part else 0
        after_cut = 1 if pv != dst_part else 0
        delta += w * (after_cut - before_cut)
    return delta


def repair_capacity(und_adj, part_of, capacity, nparts):
    """
    Enforce strict max size per part by moving boundary nodes greedily
    with minimal cut increase.
    """
    n = len(part_of)
    sizes = [0] * nparts
    for p in part_of:
        sizes[p] += 1

    overloaded = deque([p for p in range(nparts) if sizes[p] > capacity])
    if not overloaded:
        return part_of

    boundary = compute_boundary_nodes(und_adj, part_of)
    underfull = set([p for p in range(nparts) if sizes[p] < capacity])

    if not underfull and overloaded:
        raise RuntimeError("Capacity infeasible: all parts are at capacity but at least one is overloaded.")

    while overloaded:
        p_over = overloaded.popleft()
        if sizes[p_over] <= capacity:
            continue

        candidates = []
        for u in list(boundary):
            if part_of[u] != p_over:
                continue
            for p_to in underfull:
                d = move_cost_delta(und_adj, u, p_over, p_to, part_of)
                candidates.append((d, u, p_to))

        if not candidates:
            for u in range(n):
                if part_of[u] != p_over:
                    continue
                for p_to in underfull:
                    d = move_cost_delta(und_adj, u, p_over, p_to, part_of)
                    candidates.append((d, u, p_to))

        if not candidates:
            raise RuntimeError("Repair failed: no candidates to move from an overloaded part.")

        candidates.sort(key=lambda x: x[0])
        _, u_best, p_to_best = candidates[0]

        part_of[u_best] = p_to_best
        sizes[p_over] -= 1
        sizes[p_to_best] += 1

        if sizes[p_to_best] >= capacity:
            underfull.discard(p_to_best)

        boundary.add(u_best)
        for v in und_adj[u_best].keys():
            boundary.add(v)

        if sizes[p_over] > capacity:
            overloaded.append(p_over)

        if not underfull and any(s > capacity for s in sizes):
            raise RuntimeError("Capacity infeasible during repair: not enough slack to fix overload.")

    return part_of


def induced_subgraph(und_adj, nodes):
    """
    Remap node ids in 'nodes' to 0..k-1 for partitioning.
    Returns:
      sub_adj (list of dicts), old_to_new, new_to_old
    """
    nodes = sorted(nodes)
    old_to_new = {old: i for i, old in enumerate(nodes)}
    new_to_old = {i: old for old, i in old_to_new.items()}

    k = len(nodes)
    sub = [defaultdict(float) for _ in range(k)]
    for old_u in nodes:
        u = old_to_new[old_u]
        for old_v, w in und_adj[old_u].items():
            if old_v in old_to_new:
                v = old_to_new[old_v]
                if u != v and w > 0:
                    sub[u][v] += w
    return sub, old_to_new, new_to_old


def soft_rebalance_parts(sub_adj, part_of, target_size, tol, max_moves=2000):
    """
    Softly reduce oversized parts (size > target_size*(1+tol)) by moving boundary nodes
    to undersized parts, choosing moves with minimal cut increase.
    This does NOT guarantee strict enforcement, it just nudges.

    part ids are 0..P-1.
    """
    nparts = max(part_of) + 1 if part_of else 0
    if nparts <= 1:
        return part_of

    n = len(part_of)
    sizes = [0] * nparts
    for p in part_of:
        sizes[p] += 1

    hi = int(math.floor(target_size * (1.0 + tol)))
    lo = int(math.ceil(target_size * (1.0 - tol)))

    boundary = compute_boundary_nodes(sub_adj, part_of)

    moves = 0
    while moves < max_moves:
        oversized = [p for p in range(nparts) if sizes[p] > hi]
        undersized = [p for p in range(nparts) if sizes[p] < lo]
        if not oversized or not undersized:
            break

        # Work on the most oversized part first
        p_over = max(oversized, key=lambda p: sizes[p])

        best = None  # (delta_cut, u, p_to)
        # candidates: boundary nodes in p_over
        for u in list(boundary):
            if part_of[u] != p_over:
                continue
            for p_to in undersized:
                d = move_cost_delta(sub_adj, u, p_over, p_to, part_of)
                if best is None or d < best[0]:
                    best = (d, u, p_to)

        if best is None:
            break

        _, u_best, p_to_best = best
        part_of[u_best] = p_to_best
        sizes[p_over] -= 1
        sizes[p_to_best] += 1

        boundary.add(u_best)
        for v in sub_adj[u_best].keys():
            boundary.add(v)

        moves += 1

    return part_of


def partition_two_level(
    und_adj,
    num_machines,
    nodes_per_machine,
    nodes_per_worker_target,
    workers_per_machine=None,
    worker_size_tol=0.10,
    seed=42,
):
    n = len(und_adj)
    if num_machines * nodes_per_machine < n:
        raise RuntimeError("Infeasible: NUM_MACHINES * NODES_PER_MACHINE < total nodes.")

    # Stage 1: machines (STRICT cap)
    machine_of = metis_partition(und_adj, num_machines, seed=seed)
    machine_of = repair_capacity(und_adj, machine_of, nodes_per_machine, num_machines)

    nodes_in_machine = [[] for _ in range(num_machines)]
    for u, m in enumerate(machine_of):
        nodes_in_machine[m].append(u)

    # Stage 2: workers (SOFT sizing)
    worker_of_global = [-1] * n
    worker_count_per_machine = [0] * num_machines

    for m in range(num_machines):
        nodes_m = nodes_in_machine[m]
        if not nodes_m:
            continue

        k = len(nodes_m)

        if workers_per_machine is not None:
            num_workers = int(workers_per_machine)
            if num_workers <= 0:
                num_workers = 1
            target = int(math.ceil(k / float(num_workers))) if num_workers > 0 else k
        else:
            # auto choose based on target
            num_workers = int(math.ceil(k / float(nodes_per_worker_target)))
            if num_workers <= 0:
                num_workers = 1
            target = int(nodes_per_worker_target)

        sub_adj, _, new_to_old = induced_subgraph(und_adj, nodes_m)

        parts_sub = metis_partition(sub_adj, num_workers, seed=seed + m + 1)

        # Soft rebalance only: no strict capacity repair
        if num_workers > 1 and target > 0 and worker_size_tol is not None:
            parts_sub = soft_rebalance_parts(
                sub_adj=sub_adj,
                part_of=parts_sub,
                target_size=target,
                tol=float(worker_size_tol),
                max_moves=2000,
            )

        for new_u, w in enumerate(parts_sub):
            old_u = new_to_old[new_u]
            worker_of_global[old_u] = int(w)

        worker_count_per_machine[m] = num_workers

    return machine_of, worker_of_global, worker_count_per_machine


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


def compute_size_stats(part_of, nparts):
    sizes = [0] * nparts
    for p in part_of:
        sizes[p] += 1
    return {
        "min": int(min(sizes)) if sizes else 0,
        "max": int(max(sizes)) if sizes else 0,
        "sizes": sizes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to comm traces (json dict-of-dicts or edgelist)")
    ap.add_argument("--format", choices=["json", "edgelist"], required=True)
    ap.add_argument("--directed", action="store_true", help="Treat edgelist as directed (default for edgelist)")
    ap.add_argument("--num_nodes", type=int, default=None, help="Optional explicit N. Otherwise inferred.")
    ap.add_argument("--num_machines", type=int, required=True)
    ap.add_argument("--nodes_per_machine", type=int, required=True)

    ap.add_argument("--nodes_per_worker", type=int, required=True, help="Target nodes per worker (soft)")
    ap.add_argument("--workers_per_machine", type=int, default=None, help="Optional fixed workers per machine")
    ap.add_argument("--worker_size_tol", type=float, default=0.10, help="Allow +/- tol around target (default 0.10)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="Output JSON mapping node -> machine/worker")
    args = ap.parse_args()

    if args.format == "json":
        comm = load_comm_json(args.input)
    else:
        comm = load_comm_edgelist(args.input, directed=args.directed)

    n = infer_num_nodes(comm, explicit_n=args.num_nodes)
    und_adj = symmetrize_to_undirected(comm, n)

    machine_of, worker_of, worker_counts = partition_two_level(
        und_adj,
        num_machines=args.num_machines,
        nodes_per_machine=args.nodes_per_machine,
        nodes_per_worker_target=args.nodes_per_worker,
        workers_per_machine=args.workers_per_machine,
        worker_size_tol=args.worker_size_tol,
        seed=args.seed,
    )

    out = {}
    for u in range(n):
        out[str(u)] = {"machine": int(machine_of[u]), "worker": int(worker_of[u])}

    stats = {
        "num_nodes": n,
        "num_machines": args.num_machines,
        "nodes_per_machine": args.nodes_per_machine,
        "nodes_per_worker_target": args.nodes_per_worker,
        "workers_per_machine": args.workers_per_machine,
        "worker_size_tol": args.worker_size_tol,
        "machine_cut_weight": compute_cut_weight(und_adj, machine_of),
        "workers_per_machine_counts": worker_counts,
        "machine_size_stats": compute_size_stats(machine_of, args.num_machines),
    }

    payload = {"assignment": out, "stats": stats}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()