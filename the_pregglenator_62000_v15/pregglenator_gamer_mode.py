"""
pregglenator 62000 v16 - two-level partitioning (machines then workers)
Goal upgrade:
  Stage 1 (machines): minimize MAX per-machine external (network) load
    where external load of a machine is sum of cut-edge weights incident to it.
  Stage 2 (workers inside each machine): minimize MAX per-worker cross-worker load
    within a machine (same definition but on the induced subgraph).

Notes:
- Uses METIS (pymetis) for initialization, then does bottleneck-aware repair/refinement.
- Strict capacity enforced (nodes_per_machine, nodes_per_worker).
- No supersteps: comm is treated as one aggregated step.

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
    xadj, adjncy, eweights = build_pymetis_inputs(und_adj)

    # pymetis ignores seed; we keep it in signature for compatibility
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


# -------------------------
# Bottleneck (max-load) model
# -------------------------

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
    Compute how the per-part external loads change if u moves src->dst.
    Returns dict {part_id: delta_load}.
    Only parts affected: src_part, dst_part, and neighbor parts.

    Rule for an edge (u,v) with weight w:
      - Before move, it's cut if part(v) != src
      - After move, it's cut if part(v) != dst

    Each cut edge contributes w to both endpoint parts.
    So changing cut-status changes loads for:
      - the part of u (src or dst) by +/- w
      - the part of v by +/- w
    """
    deltas = defaultdict(float)

    for v, w in und_adj[u].items():
        pv = part_of[v]

        before_cut = (pv != src_part)
        after_cut = (pv != dst_part)

        if before_cut == after_cut:
            continue

        if before_cut and (not after_cut):
            # edge was cut, becomes internal: remove contribution
            deltas[src_part] -= w
            deltas[pv] -= w
        else:
            # edge was internal, becomes cut: add contribution
            deltas[dst_part] += w
            deltas[pv] += w

    # Note: u's membership changes, so any "cut edge" contributions incident to u
    # are already accounted via src_part/dst_part adjustments above.

    return deltas


def score_max_load_with_deltas(loads, deltas):
    """
    Given current loads list and sparse deltas dict, compute new max(loads).
    """
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
    underfull_parts,
    boundary_nodes,
    nparts,
):
    """
    Pick a move u: src_part -> dst_part that minimizes resulting max external load.
    Only considers u in boundary_nodes (or falls back to all nodes in src_part).
    Respects capacity on dst_part.

    Returns (best_u, best_dst, best_new_max, best_deltas) or (None,...)
    """
    current_max = max(loads)

    candidates_u = [u for u in boundary_nodes if part_of[u] == src_part]
    if not candidates_u:
        candidates_u = [u for u in range(len(part_of)) if part_of[u] == src_part]

    best = None  # (new_max, tie_cut_delta, u, dst, deltas)
    for u in candidates_u:
        for dst in underfull_parts:
            if dst == src_part:
                continue
            if sizes[dst] >= capacity:
                continue

            deltas = move_delta_external_loads(und_adj, u, src_part, dst, part_of)
            new_max = score_max_load_with_deltas(loads, deltas)

            # Tie-break: if max is same, prefer not increasing total cut too much.
            # Approx: delta_total_cut = 0.5 * (delta_load_src + delta_load_dst + sum neighbor deltas? not exact)
            # We'll use simple: sum of positive deltas - sum of negative deltas over all parts, scaled.
            # This is just a stable tiebreaker.
            tie = 0.0
            for dv in deltas.values():
                tie += abs(dv)

            if best is None:
                best = (new_max, tie, u, dst, deltas)
                continue

            if new_max < best[0] - 1e-12:
                best = (new_max, tie, u, dst, deltas)
            elif abs(new_max - best[0]) <= 1e-12 and tie < best[1]:
                best = (new_max, tie, u, dst, deltas)

    if best is None:
        return None, None, None, None

    # If we can't improve max at all, still may need a move for capacity repair.
    # Caller decides whether to accept.
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
    Enforce strict capacity using bottleneck objective:
      - Primary: minimize resulting max external load (max per-part).
      - Secondary: stable small changes.

    If refine_iters > 0, also performs extra refinement moves after capacity is feasible.
    """
    n = len(part_of)
    sizes = compute_part_sizes(part_of, nparts)

    if all(s <= capacity for s in sizes):
        loads = compute_part_external_loads(und_adj, part_of, nparts)
        if refine_iters > 0:
            _refine_bottleneck(und_adj, part_of, loads, sizes, capacity, nparts, refine_iters)
        return part_of

    # Need slack somewhere
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

        # Always try to move out of src to any underfull dst with best bottleneck score
        u, dst, new_max, deltas = choose_best_bottleneck_move(
            und_adj=und_adj,
            part_of=part_of,
            loads=loads,
            sizes=sizes,
            capacity=capacity,
            src_part=src,
            underfull_parts=underfull,
            boundary_nodes=boundary,
            nparts=nparts,
        )

        if u is None:
            raise RuntimeError("Repair failed: no candidates to move from an overloaded part.")

        apply_move(part_of, loads, sizes, u, src, dst, deltas)

        if sizes[dst] >= capacity:
            underfull.discard(dst)

        # boundary update local
        boundary.add(u)
        for v in und_adj[u].keys():
            boundary.add(v)

        if sizes[src] > capacity:
            overloaded.append(src)

        if not underfull and any(s > capacity for s in sizes):
            raise RuntimeError("Capacity infeasible during repair: not enough slack to fix overload.")

    if refine_iters > 0:
        _refine_bottleneck(und_adj, part_of, loads, sizes, capacity, nparts, refine_iters)

    return part_of


def _refine_bottleneck(und_adj, part_of, loads, sizes, capacity, nparts, iters):
    """
    Local refinement to reduce max external load while respecting capacity.
    Greedy: repeatedly pick worst-loaded part and move a boundary node out if it reduces max.
    """
    boundary = compute_boundary_nodes(und_adj, part_of)

    for _ in range(iters):
        worst = max(range(nparts), key=lambda p: loads[p])
        current_max = loads[worst]

        # Candidate destinations: any part with slack
        dsts = [p for p in range(nparts) if p != worst and sizes[p] < capacity]
        if not dsts:
            break

        u, dst, new_max, deltas = choose_best_bottleneck_move(
            und_adj=und_adj,
            part_of=part_of,
            loads=loads,
            sizes=sizes,
            capacity=capacity,
            src_part=worst,
            underfull_parts=set(dsts),
            boundary_nodes=boundary,
            nparts=nparts,
        )

        if u is None:
            break

        # Only accept if it improves the global max
        if new_max >= max(loads) - 1e-12:
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


def partition_two_level_bottleneck(
    und_adj,
    num_machines,
    nodes_per_machine,
    nodes_per_worker,
    seed=42,
    refine_machine_iters=200,
    refine_worker_iters=100,
):
    n = len(und_adj)
    if num_machines * nodes_per_machine < n:
        raise RuntimeError("Infeasible: NUM_MACHINES * NODES_PER_MACHINE < total nodes.")

    # Stage 1: machines init (METIS)
    machine_of = metis_partition(und_adj, num_machines, seed=seed)

    # Bottleneck-aware strict capacity repair + refinement
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

    # Stage 2: workers inside each machine
    worker_of_global = [-1] * n
    worker_count_per_machine = [0] * num_machines

    for m in range(num_machines):
        nodes_m = nodes_in_machine[m]
        if not nodes_m:
            continue

        k = len(nodes_m)
        num_workers = int(math.ceil(k / float(nodes_per_worker)))
        if num_workers <= 0:
            num_workers = 1

        sub_adj, _, new_to_old = induced_subgraph(und_adj, nodes_m)

        # init workers with METIS on induced subgraph
        parts_sub = metis_partition(sub_adj, num_workers, seed=seed + m + 1)

        # capacity repair on workers, bottleneck objective WITHIN machine
        parts_sub = repair_capacity_bottleneck(
            sub_adj,
            parts_sub,
            capacity=nodes_per_worker,
            nparts=num_workers,
            refine_iters=refine_worker_iters,
        )

        for new_u, w in enumerate(parts_sub):
            old_u = new_to_old[new_u]
            worker_of_global[old_u] = w

        worker_count_per_machine[m] = num_workers

    return machine_of, worker_of_global, worker_count_per_machine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to comm traces (json dict-of-dicts or edgelist)")
    ap.add_argument("--format", choices=["json", "edgelist"], required=True)
    ap.add_argument("--directed", action="store_true", help="Treat edgelist as directed (default for edgelist)")
    ap.add_argument("--num_nodes", type=int, default=None, help="Optional explicit N. Otherwise inferred.")
    ap.add_argument("--num_machines", type=int, required=True)
    ap.add_argument("--num_workers", type=int, default=4, help="Workers per machine.")
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

    nodes_per_machine = math.ceil(n / args.num_machines)
    nodes_per_worker = math.ceil(nodes_per_machine / args.num_workers)
    print(f"Inferred nodes_per_machine: {nodes_per_machine}, nodes_per_worker: {nodes_per_worker}")

    machine_of, worker_of, worker_counts = partition_two_level_bottleneck(
        und_adj,
        num_machines=args.num_machines,
        nodes_per_machine=nodes_per_machine,
        nodes_per_worker=nodes_per_worker,
        seed=args.seed,
        refine_machine_iters=args.refine_machine_iters,
        refine_worker_iters=args.refine_worker_iters,
    )

    # Build output
    out = {}
    for u in range(n):
        out[str(u)] = {"machine": int(machine_of[u]), "worker": int(worker_of[u])}

    machine_cut = compute_cut_weight(und_adj, machine_of)
    max_load, per_loads = compute_max_external_load(und_adj, machine_of, args.num_machines)

    stats = {
        "num_nodes": n,
        "num_machines": args.num_machines,
        "nodes_per_machine": nodes_per_machine,
        "nodes_per_worker": nodes_per_worker,
        "machine_cut_weight": machine_cut,
        "machine_max_external_load": max_load,
        "machine_external_loads": per_loads,
        "workers_per_machine": worker_counts,
    }

    payload = {"assignment": out, "stats": stats}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()