"""
pregglenator 62000 v17 - two-level partitioning with HARD workers-per-machine

Stage 1 (machines):
  - Init with METIS
  - Repair + optional refine with bottleneck objective: minimize MAX per-machine external load
  - Enforce strict nodes_per_machine (upper bound)

Stage 2 (workers inside each machine):
  - HARD enforce workers_per_machine = W (exactly)
  - Enforce strict nodes_per_worker (upper bound)
  - Repair + optional refine with bottleneck objective on induced subgraph

If any machine ends up with k nodes where k > W * nodes_per_worker, stage 2 is infeasible
and the script raises an error.

Install:
  conda install -c conda-forge pymetis
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
    # pymetis doesn't expose a seed; keep arg for compatibility
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


def compute_part_sizes(part_of, nparts):
    sizes = [0] * nparts
    for p in part_of:
        sizes[p] += 1
    return sizes


def compute_part_external_loads(und_adj, part_of, nparts):
    """
    External load per part = sum of cut-edge weights incident to that part.
    Each cut edge contributes its weight to BOTH endpoint parts.
    """
    loads = [0.0] * nparts
    seen = set()
    n = len(und_adj)
    for u in range(n):
        pu = part_of[u]
        for v, w in und_adj[u].items():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            pv = part_of[v]
            if pu != pv:
                loads[pu] += w
                loads[pv] += w
    return loads


def move_delta_external_loads(und_adj, u, src_part, dst_part, part_of):
    """
    Sparse per-part load deltas if u moves src->dst, under the external-load definition.
    """
    deltas = defaultdict(float)

    for v, w in und_adj[u].items():
        pv = part_of[v]
        before_cut = (pv != src_part)
        after_cut = (pv != dst_part)

        if before_cut == after_cut:
            continue

        if before_cut and (not after_cut):
            # cut -> internal: remove w from both endpoint parts
            deltas[src_part] -= w
            deltas[pv] -= w
        else:
            # internal -> cut: add w to both endpoint parts
            deltas[dst_part] += w
            deltas[pv] += w

    return deltas


def score_max_load_with_deltas(loads, deltas):
    mx = -1.0
    for i, base in enumerate(loads):
        val = base + deltas.get(i, 0.0)
        if val > mx:
            mx = val
    return mx


def choose_best_bottleneck_move(
    und_adj,
    part_of,
    loads,
    sizes,
    capacity,
    src_part,
    candidate_dsts,
    boundary_nodes,
    nparts,
):
    """
    Pick u in src_part and dst in candidate_dsts minimizing resulting max load.
    Returns (u, dst, new_max, deltas) or (None,...)
    """
    candidates_u = [u for u in boundary_nodes if part_of[u] == src_part]
    if not candidates_u:
        candidates_u = [u for u in range(len(part_of)) if part_of[u] == src_part]

    best = None  # (new_max, tie, u, dst, deltas)
    for u in candidates_u:
        for dst in candidate_dsts:
            if dst == src_part:
                continue
            if sizes[dst] >= capacity:
                continue

            deltas = move_delta_external_loads(und_adj, u, src_part, dst, part_of)
            new_max = score_max_load_with_deltas(loads, deltas)

            tie = 0.0
            for dv in deltas.values():
                tie += abs(dv)

            if best is None:
                best = (new_max, tie, u, dst, deltas)
            else:
                if new_max < best[0] - 1e-12:
                    best = (new_max, tie, u, dst, deltas)
                elif abs(new_max - best[0]) <= 1e-12 and tie < best[1]:
                    best = (new_max, tie, u, dst, deltas)

    if best is None:
        return None, None, None, None
    return best[2], best[3], best[0], best[4]


def apply_move(part_of, loads, sizes, u, src, dst, deltas):
    part_of[u] = dst
    sizes[src] -= 1
    sizes[dst] += 1
    for p, d in deltas.items():
        loads[p] += d


def repair_capacity_bottleneck(
    und_adj,
    part_of,
    capacity,
    nparts,
    refine_iters=0,
):
    """
    Enforce strict max size per part by moving nodes, choosing moves that minimize
    resulting max external load (bottleneck objective). Then optionally refine.
    """
    sizes = compute_part_sizes(part_of, nparts)

    # Quick feasible case
    if all(s <= capacity for s in sizes):
        loads = compute_part_external_loads(und_adj, part_of, nparts)
        if refine_iters > 0:
            refine_bottleneck(und_adj, part_of, loads, sizes, capacity, nparts, refine_iters)
        return part_of

    underfull = set([p for p in range(nparts) if sizes[p] < capacity])
    overloaded = deque([p for p in range(nparts) if sizes[p] > capacity])

    if not underfull and overloaded:
        raise RuntimeError("Capacity infeasible: all parts are at capacity but at least one is overloaded.")

    boundary = compute_boundary_nodes(und_adj, part_of)
    loads = compute_part_external_loads(und_adj, part_of, nparts)

    while overloaded:
        src = overloaded.popleft()
        if sizes[src] <= capacity:
            continue

        u, dst, _, deltas = choose_best_bottleneck_move(
            und_adj=und_adj,
            part_of=part_of,
            loads=loads,
            sizes=sizes,
            capacity=capacity,
            src_part=src,
            candidate_dsts=underfull,
            boundary_nodes=boundary,
            nparts=nparts,
        )

        if u is None:
            raise RuntimeError("Repair failed: no candidates to move from an overloaded part.")

        apply_move(part_of, loads, sizes, u, src, dst, deltas)

        if sizes[dst] >= capacity:
            underfull.discard(dst)

        boundary.add(u)
        for v in und_adj[u].keys():
            boundary.add(v)

        if sizes[src] > capacity:
            overloaded.append(src)

        if not underfull and any(s > capacity for s in sizes):
            raise RuntimeError("Capacity infeasible during repair: not enough slack to fix overload.")

    if refine_iters > 0:
        refine_bottleneck(und_adj, part_of, loads, sizes, capacity, nparts, refine_iters)

    return part_of


def refine_bottleneck(und_adj, part_of, loads, sizes, capacity, nparts, iters):
    """
    Greedy refinement: repeatedly take worst-loaded part and attempt a move out
    that improves global max load.
    """
    boundary = compute_boundary_nodes(und_adj, part_of)

    for _ in range(iters):
        worst = max(range(nparts), key=lambda p: loads[p])
        current_max = max(loads)

        candidate_dsts = [p for p in range(nparts) if p != worst and sizes[p] < capacity]
        if not candidate_dsts:
            break

        u, dst, new_max, deltas = choose_best_bottleneck_move(
            und_adj=und_adj,
            part_of=part_of,
            loads=loads,
            sizes=sizes,
            capacity=capacity,
            src_part=worst,
            candidate_dsts=candidate_dsts,
            boundary_nodes=boundary,
            nparts=nparts,
        )
        if u is None:
            break
        if new_max >= current_max - 1e-12:
            break

        apply_move(part_of, loads, sizes, u, worst, dst, deltas)

        boundary.add(u)
        for v in und_adj[u].keys():
            boundary.add(v)


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


def compute_max_external_load(und_adj, part_of, nparts):
    loads = compute_part_external_loads(und_adj, part_of, nparts)
    return max(loads), loads


def partition_two_level_fixed_workers(
    und_adj,
    num_machines,
    nodes_per_machine,
    workers_per_machine,
    nodes_per_worker,
    seed=42,
    refine_machine_iters=200,
    refine_worker_iters=100,
):
    n = len(und_adj)
    if num_machines * nodes_per_machine < n:
        raise RuntimeError("Infeasible: NUM_MACHINES * NODES_PER_MACHINE < total nodes.")

    # Stage 1: machine partition init + bottleneck repair/refine
    machine_of = metis_partition(und_adj, num_machines, seed=seed)
    machine_of = repair_capacity_bottleneck(
        und_adj,
        machine_of,
        capacity=nodes_per_machine,
        nparts=num_machines,
        refine_iters=refine_machine_iters,
    )

    # Collect nodes per machine
    nodes_in_machine = [[] for _ in range(num_machines)]
    for u, m in enumerate(machine_of):
        nodes_in_machine[m].append(u)

    # Stage 2: fixed workers per machine
    worker_of_global = [-1] * n
    worker_count_per_machine = [workers_per_machine] * num_machines

    for m in range(num_machines):
        nodes_m = nodes_in_machine[m]
        k = len(nodes_m)
        if k == 0:
            continue

        if k > workers_per_machine * nodes_per_worker:
            raise RuntimeError(
                f"Workers infeasible on machine {m}: "
                f"{k} nodes but workers_per_machine*nodes_per_worker="
                f"{workers_per_machine}*{nodes_per_worker}={workers_per_machine*nodes_per_worker}."
            )

        sub_adj, _, new_to_old = induced_subgraph(und_adj, nodes_m)

        # If only 1 worker, trivial
        if workers_per_machine == 1:
            for new_u in range(len(nodes_m)):
                old_u = new_to_old[new_u]
                worker_of_global[old_u] = 0
            continue

        parts_sub = metis_partition(sub_adj, workers_per_machine, seed=seed + m + 1)

        parts_sub = repair_capacity_bottleneck(
            sub_adj,
            parts_sub,
            capacity=nodes_per_worker,
            nparts=workers_per_machine,
            refine_iters=refine_worker_iters,
        )

        for new_u, w in enumerate(parts_sub):
            old_u = new_to_old[new_u]
            worker_of_global[old_u] = w

    return machine_of, worker_of_global, worker_count_per_machine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to comm traces (json dict-of-dicts or edgelist)")
    ap.add_argument("--format", choices=["json", "edgelist"], required=True)
    ap.add_argument("--directed", action="store_true", help="Treat edgelist as directed (default for edgelist)")
    ap.add_argument("--num_nodes", type=int, default=None, help="Optional explicit N. Otherwise inferred.")
    ap.add_argument("--num_machines", type=int, required=True)
    ap.add_argument("--nodes_per_machine", type=int, required=True)

    ap.add_argument("--workers_per_machine", type=int, required=True, help="HARD exact worker count per machine")
    ap.add_argument("--nodes_per_worker", type=int, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refine_machine_iters", type=int, default=200)
    ap.add_argument("--refine_worker_iters", type=int, default=100)
    ap.add_argument("--output", required=True, help="Output JSON mapping node -> machine/worker")
    args = ap.parse_args()

    if args.format == "json":
        comm = load_comm_json(args.input)
    else:
        comm = load_comm_edgelist(args.input, directed=args.directed)

    n = infer_num_nodes(comm, explicit_n=args.num_nodes)
    und_adj = symmetrize_to_undirected(comm, n)

    machine_of, worker_of, worker_counts = partition_two_level_fixed_workers(
        und_adj,
        num_machines=args.num_machines,
        nodes_per_machine=args.nodes_per_machine,
        workers_per_machine=args.workers_per_machine,
        nodes_per_worker=args.nodes_per_worker,
        seed=args.seed,
        refine_machine_iters=args.refine_machine_iters,
        refine_worker_iters=args.refine_worker_iters,
    )

    out = {}
    for u in range(n):
        out[str(u)] = {"machine": int(machine_of[u]), "worker": int(worker_of[u])}

    machine_cut = compute_cut_weight(und_adj, machine_of)
    max_load, per_loads = compute_max_external_load(und_adj, machine_of, args.num_machines)

    stats = {
        "num_nodes": n,
        "num_machines": args.num_machines,
        "nodes_per_machine": args.nodes_per_machine,
        "workers_per_machine": args.workers_per_machine,
        "nodes_per_worker": args.nodes_per_worker,
        "machine_cut_weight": machine_cut,
        "machine_max_external_load": max_load,
        "machine_external_loads": per_loads,
        "workers_per_machine_list": worker_counts,
    }

    payload = {"assignment": out, "stats": stats}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()