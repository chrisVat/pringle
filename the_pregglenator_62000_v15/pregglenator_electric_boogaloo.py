"""
pregglenator v16 — superstep-aware partitioning with simulated annealing

Inputs : comm_traces/src_*/merged.csv
         columns: source, superstep, src_vertex, dst_vertex, count

Algorithm
---------
1. Load per-superstep communication traces (multiple SSSP source problems)
2. METIS initialisation
     edge weights   = total communication between nodes (all supersteps)
     vertex weights = activity breadth (distinct supersteps vertex is active in)
3. Simulated annealing refines machine assignment against the unified cost:
       c_net  × cross-machine messages  (summed over all supersteps)
     + c_node × Σ_s  max_machine( compute_load(machine, s) )
   where compute_load(m, s) = messages received by vertices on m in superstep s
   c_proc is excluded from SA because worker assignment hasn't happened yet.
4. METIS assigns workers within each machine on the induced subgraph.
5. Report per-superstep utilisation and the full three-component cost in stats.

Output JSON (same schema as v15)
  { "assignment": { "<node_id>": { "machine": int, "worker": int }, ... },
    "stats": { ... } }
"""

import argparse
import csv
import glob
import json
import math
import os
import random
from collections import defaultdict, deque

import pymetis


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_comm_traces(traces_dir):
    """
    Find every src_*/merged.csv under *traces_dir* and aggregate.

    Returns
    -------
    comm_by_superstep : {s: {(u, v): total_count}}
    total_comm        : {(u, v): total_count_across_all_supersteps}
    recv_by_vertex    : {v: {s: total_msgs_received_in_s}}
    """
    paths = sorted(glob.glob(os.path.join(traces_dir, "src_*", "merged.csv")))
    if not paths:
        raise RuntimeError(
            f"No merged.csv files found under {traces_dir}/src_*/\n"
            "Expected layout: <traces_dir>/src_<N>/merged.csv"
        )
    print(f"  {len(paths)} trace file(s): "
          f"{[os.path.basename(os.path.dirname(p)) for p in paths]}")

    comm_ss = defaultdict(lambda: defaultdict(float))  # s → {(u,v): c}
    recv_vx = defaultdict(lambda: defaultdict(float))  # v → {s: c}

    for path in paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                s = int(row["superstep"])
                u = int(row["src_vertex"])
                v = int(row["dst_vertex"])
                c = float(row["count"])
                comm_ss[s][(u, v)] += c
                recv_vx[v][s]      += c

    total_comm = defaultdict(float)
    for edges in comm_ss.values():
        for (u, v), c in edges.items():
            total_comm[(u, v)] += c

    return (
        {s: dict(e)  for s, e  in comm_ss.items()},
        dict(total_comm),
        {v: dict(ss) for v, ss in recv_vx.items()},
    )


def infer_num_nodes(total_comm, explicit_n=None):
    if explicit_n is not None:
        return explicit_n
    mx = max((max(u, v) for u, v in total_comm), default=-1)
    return mx + 1


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def symmetrize_to_undirected(total_comm, n):
    """
    Build undirected adjacency list from *total_comm*.
    und[u][v] = w(u→v) + w(v→u)
    """
    directed = defaultdict(dict)
    for (u, v), c in total_comm.items():
        if 0 <= u < n and 0 <= v < n and u != v:
            directed[u][v] = directed[u].get(v, 0.0) + c

    und  = [defaultdict(float) for _ in range(n)]
    seen = set()
    for u, nbrs in directed.items():
        for v, w_uv in nbrs.items():
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            w_vu = directed.get(v, {}).get(u, 0.0)
            w    = w_uv + w_vu
            if w > 0:
                und[u][v] = w
                und[v][u] = w
    return und


def build_activity_vweights(recv_by_vertex, n):
    """vweight[v] = distinct supersteps in which v receives at least one message."""
    w = [1] * n   # floor at 1 so METIS never sees a zero-weight node
    for v, ss in recv_by_vertex.items():
        if v < n:
            w[v] = max(1, len(ss))
    return w


# ─────────────────────────────────────────────────────────────────────────────
# 3.  METIS wrappers  (adapted from v15)
# ─────────────────────────────────────────────────────────────────────────────

def _build_pymetis_inputs(und_adj):
    n        = len(und_adj)
    xadj     = [0]
    adjncy   = []
    eweights = []
    for u in range(n):
        for v, w in sorted(und_adj[u].items()):
            adjncy.append(v)
            eweights.append(max(1, int(round(w))))
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights


def metis_partition(und_adj, nparts, vweights=None, seed=42):
    xadj, adjncy, eweights = _build_pymetis_inputs(und_adj)
    adjacency = pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy)
    kwargs = dict(adjacency=adjacency, eweights=eweights)
    if vweights is not None:
        kwargs["vweights"] = [max(1, int(v)) for v in vweights]
    result = pymetis.part_graph(nparts, **kwargs)
    return list(result.vertex_part)


def _compute_boundary(und_adj, part_of):
    boundary = set()
    for u in range(len(und_adj)):
        for v in und_adj[u]:
            if part_of[v] != part_of[u]:
                boundary.add(u)
                break
    return boundary


def _move_delta_cut(und_adj, u, src, dst, part_of):
    delta = 0.0
    for v, w in und_adj[u].items():
        pv     = part_of[v]
        delta += w * ((1 if pv != dst else 0) - (1 if pv != src else 0))
    return delta


def repair_capacity(und_adj, part_of, capacity, nparts):
    """Enforce strict ≤ capacity per part by moving boundary nodes greedily."""
    n     = len(part_of)
    sizes = [0] * nparts
    for p in part_of:
        sizes[p] += 1

    overloaded = deque(p for p in range(nparts) if sizes[p] > capacity)
    if not overloaded:
        return part_of

    boundary  = _compute_boundary(und_adj, part_of)
    underfull = set(p for p in range(nparts) if sizes[p] < capacity)
    if not underfull:
        raise RuntimeError("Capacity infeasible: all parts full but some overloaded.")

    while overloaded:
        p_over = overloaded.popleft()
        if sizes[p_over] <= capacity:
            continue

        cands = [
            (_move_delta_cut(und_adj, u, p_over, p_to, part_of), u, p_to)
            for u  in boundary if part_of[u] == p_over
            for p_to in underfull
        ]
        if not cands:
            cands = [
                (_move_delta_cut(und_adj, u, p_over, p_to, part_of), u, p_to)
                for u  in range(n) if part_of[u] == p_over
                for p_to in underfull
            ]
        if not cands:
            raise RuntimeError("Repair failed: no moveable nodes.")

        cands.sort()
        _, u_best, p_to_best = cands[0]
        part_of[u_best]  = p_to_best
        sizes[p_over]   -= 1
        sizes[p_to_best] += 1

        if sizes[p_to_best] >= capacity:
            underfull.discard(p_to_best)

        boundary.add(u_best)
        for nb in und_adj[u_best]:
            boundary.add(nb)

        if sizes[p_over] > capacity:
            overloaded.append(p_over)

        if not underfull and any(s > capacity for s in sizes):
            raise RuntimeError("Capacity infeasible during repair.")

    return part_of


def induced_subgraph(und_adj, nodes):
    nodes      = sorted(nodes)
    old_to_new = {old: i for i, old in enumerate(nodes)}
    k          = len(nodes)
    sub        = [defaultdict(float) for _ in range(k)]
    for old_u in nodes:
        u = old_to_new[old_u]
        for old_v, w in und_adj[old_u].items():
            if old_v in old_to_new:
                v = old_to_new[old_v]
                if u != v and w > 0:
                    sub[u][v] += w
    return sub, nodes   # nodes[new_id] = old_id


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SA edge index and cost helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_vertex_edge_index(comm_by_superstep, n):
    """
    Flat per-vertex send / receive lists for O(degree) delta computation.
      send_edges[u] = [(s, dst, count), ...]
      recv_edges[v] = [(s, src, count), ...]
    """
    send_edges = defaultdict(list)
    recv_edges = defaultdict(list)
    for s, edges in comm_by_superstep.items():
        for (u, v), c in edges.items():
            if 0 <= u < n and 0 <= v < n:
                send_edges[u].append((s, v, c))
                recv_edges[v].append((s, u, c))
    return dict(send_edges), dict(recv_edges)


def _initial_machine_loads(machine_of, recv_by_vertex, num_machines, supersteps, n):
    """ml[s][m] = total recv-msgs on machine m in superstep s."""
    ml = {s: [0.0] * num_machines for s in supersteps}
    for v, ss in recv_by_vertex.items():
        if v < n:
            m = machine_of[v]
            for s, load in ss.items():
                if s in ml:
                    ml[s][m] += load
    return ml


def _sample_initial_temperature(
    machine_of, send_edges, recv_edges, recv_by_vertex,
    ml, supersteps, num_machines, sizes, nodes_per_machine,
    n, c_net, c_node, rng, k=1000
):
    """Estimate T_start as 2 × average |Δcost| over k random moves."""
    deltas   = []
    vertices = list(range(n))
    for _ in range(k):
        v      = rng.choice(vertices)
        m_from = machine_of[v]
        cands  = [m for m in range(num_machines)
                  if m != m_from and sizes[m] < nodes_per_machine]
        if not cands:
            continue
        m_to = rng.choice(cands)

        d = 0.0
        for _, dst, c in send_edges.get(v, []):
            md  = machine_of[dst]
            d  += c_net * c * ((0 if md == m_to else 1) - (0 if md == m_from else 1))
        for _, src, c in recv_edges.get(v, []):
            ms  = machine_of[src]
            d  += c_net * c * ((0 if ms == m_to else 1) - (0 if ms == m_from else 1))

        v_recv = recv_by_vertex.get(v, {})
        for s in supersteps:
            rv = v_recv.get(s, 0.0)
            if rv:
                old_max        = max(ml[s])
                ml[s][m_from] -= rv
                ml[s][m_to]   += rv
                new_max        = max(ml[s])
                ml[s][m_from] += rv   # restore
                ml[s][m_to]   -= rv
                d += c_node * (new_max - old_max)

        deltas.append(abs(d))

    avg = sum(deltas) / len(deltas) if deltas else 1.0
    return max(avg * 2.0, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Simulated annealing
# ─────────────────────────────────────────────────────────────────────────────

def sa_optimize_machines(
    machine_of, send_edges, recv_edges, recv_by_vertex,
    num_machines, nodes_per_machine, supersteps, n,
    c_net, c_node, t_start, t_end, sa_iters, seed=42
):
    """
    Refine machine assignment via SA.

    Optimises
    ---------
      cost = c_net  × Σ cross-machine message counts (all supersteps)
           + c_node × Σ_s  max_m( recv-load on machine m in superstep s )

    Capacity constraint is enforced by only proposing moves to non-full machines.
    c_proc is not included here; it is implicitly handled in the worker stage.
    """
    rng        = random.Random(seed)
    machine_of = list(machine_of)
    sizes      = [0] * num_machines
    for m in machine_of:
        sizes[m] += 1

    ml = _initial_machine_loads(machine_of, recv_by_vertex, num_machines, supersteps, n)

    if sa_iters <= 1 or t_start <= t_end:
        cooling = 1.0
    else:
        cooling = (t_end / t_start) ** (1.0 / (sa_iters - 1))

    T         = t_start
    accepted  = 0
    vertices  = list(range(n))
    log_every = max(1, sa_iters // 10)

    for iteration in range(sa_iters):
        v      = rng.choice(vertices)
        m_from = machine_of[v]

        # Only move to machines with remaining capacity
        cands = [m for m in range(num_machines)
                 if m != m_from and sizes[m] < nodes_per_machine]
        if not cands:
            T *= cooling
            continue

        m_to = rng.choice(cands)

        # ── Δ communication cost ─────────────────────────────────────────
        # Cost of edge (u→w) = c_net if machine(u) ≠ machine(w), else 0.
        # Moving v changes the machine-membership for all edges incident to v.
        d_comm = 0.0
        for _, dst, c in send_edges.get(v, []):
            md      = machine_of[dst]
            d_comm += c_net * c * ((0 if md == m_to else 1) -
                                   (0 if md == m_from else 1))
        for _, src, c in recv_edges.get(v, []):
            ms      = machine_of[src]
            d_comm += c_net * c * ((0 if ms == m_to else 1) -
                                   (0 if ms == m_from else 1))

        # ── Δ compute-bottleneck cost ────────────────────────────────────
        # v's recv-load shifts from m_from to m_to in every superstep it is active.
        # We temporarily modify ml, read the new max, then restore.
        d_compute = 0.0
        v_recv    = recv_by_vertex.get(v, {})
        for s in supersteps:
            rv = v_recv.get(s, 0.0)
            if rv == 0.0:
                continue
            old_max        = max(ml[s])
            ml[s][m_from] -= rv
            ml[s][m_to]   += rv
            new_max        = max(ml[s])
            ml[s][m_from] += rv   # restore
            ml[s][m_to]   -= rv
            d_compute     += c_node * (new_max - old_max)

        delta = d_comm + d_compute

        # ── Accept / reject ──────────────────────────────────────────────
        if delta < 0.0 or rng.random() < math.exp(-delta / T):
            machine_of[v]  = m_to
            sizes[m_from] -= 1
            sizes[m_to]   += 1
            for s in supersteps:
                rv = v_recv.get(s, 0.0)
                if rv:
                    ml[s][m_from] -= rv
                    ml[s][m_to]   += rv
            accepted += 1

        T *= cooling

        if (iteration + 1) % log_every == 0:
            rate = accepted / (iteration + 1)
            print(f"  [SA] {iteration+1:>8}/{sa_iters}  "
                  f"T={T:.6f}  accept_rate={rate:.3f}")

    print(f"  [SA] done — accepted {accepted}/{sa_iters} "
          f"({100*accepted/sa_iters:.1f}%)")
    return machine_of


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Worker assignment within machines
# ─────────────────────────────────────────────────────────────────────────────

def assign_workers(und_adj, machine_of, num_machines, nodes_per_worker, seed=42):
    n                   = len(und_adj)
    worker_of           = [-1] * n
    workers_per_machine = []

    for m in range(num_machines):
        nodes_m     = [u for u in range(n) if machine_of[u] == m]
        num_workers = max(1, math.ceil(len(nodes_m) / nodes_per_worker))

        if not nodes_m:
            workers_per_machine.append(0)
            continue

        if num_workers == 1:
            for u in nodes_m:
                worker_of[u] = 0
        else:
            sub_adj, new_to_old = induced_subgraph(und_adj, nodes_m)
            parts = metis_partition(sub_adj, num_workers, seed=seed + m + 1)
            parts = repair_capacity(sub_adj, parts, nodes_per_worker, num_workers)
            for new_u, w in enumerate(parts):
                worker_of[new_to_old[new_u]] = w

        workers_per_machine.append(num_workers)

    return worker_of, workers_per_machine


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_cut_weight(und_adj, part_of):
    cut  = 0.0
    seen = set()
    for u in range(len(und_adj)):
        pu = part_of[u]
        for v, w in und_adj[u].items():
            if (v, u) not in seen:
                seen.add((u, v))
                if part_of[v] != pu:
                    cut += w
    return cut


def compute_superstep_utilisation(machine_of, recv_by_vertex, num_machines, supersteps, n):
    """Returns {superstep: num_machines_with_at_least_one_active_vertex}."""
    util = {}
    for s in supersteps:
        active = set()
        for v, ss in recv_by_vertex.items():
            if s in ss and v < n:
                active.add(machine_of[v])
        util[s] = len(active)
    return util


def compute_final_cost(
    machine_of, worker_of, comm_by_superstep, recv_by_vertex,
    num_machines, supersteps, n, c_net, c_proc, c_node
):
    """
    Full three-component cost after worker assignment is known.
    Returns (comm_cost, compute_bottleneck).
    """
    comm_cost = 0.0
    for s, edges in comm_by_superstep.items():
        for (u, v), c in edges.items():
            if machine_of[u] != machine_of[v]:
                comm_cost += c_net  * c
            elif worker_of[u] != worker_of[v]:
                comm_cost += c_proc * c

    compute_bottleneck = 0.0
    for s in supersteps:
        ml = [0.0] * num_machines
        for v, ss in recv_by_vertex.items():
            if s in ss and v < n:
                ml[machine_of[v]] += ss[s]
        compute_bottleneck += c_node * max(ml)

    return comm_cost, compute_bottleneck


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Superstep-aware Pregel partitioner with SA refinement"
    )
    ap.add_argument("--traces_dir",        required=True,
                    help="Directory containing src_*/merged.csv trace files")
    ap.add_argument("--num_nodes",         type=int,   default=None,
                    help="Graph size N (inferred from data if omitted)")
    ap.add_argument("--num_machines",      type=int,   required=True)
    ap.add_argument("--nodes_per_machine", type=int,   required=True)
    ap.add_argument("--nodes_per_worker",  type=int,   required=True)
    ap.add_argument("--c_net",             type=float, default=100.0,
                    help="Cost weight for cross-machine messages (default 100)")
    ap.add_argument("--c_proc",            type=float, default=10.0,
                    help="Cost weight for cross-worker same-machine messages (default 10)")
    ap.add_argument("--c_node",            type=float, default=1.0,
                    help="Cost weight per unit of vertex compute load (default 1)")
    ap.add_argument("--sa_iters",          type=int,   default=5_000_000,
                    help="Total SA iterations (default 5 000 000)")
    ap.add_argument("--t_start",           type=float, default=None,
                    help="SA start temperature (auto-sampled if omitted)")
    ap.add_argument("--t_end",             type=float, default=1e-4,
                    help="SA end temperature (default 1e-4)")
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--output",            required=True,
                    help="Output JSON path")
    args = ap.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────
    print("Loading comm traces...")
    comm_by_superstep, total_comm, recv_by_vertex = load_comm_traces(args.traces_dir)
    supersteps = sorted(comm_by_superstep.keys())
    n_edges    = sum(len(e) for e in comm_by_superstep.values())
    print(f"  {len(supersteps)} supersteps  |  {n_edges:,} (superstep, edge) pairs")

    n = infer_num_nodes(total_comm, args.num_nodes)
    print(f"  {n:,} nodes (IDs 0–{n-1})")

    # ── Build undirected graph ────────────────────────────────────────────
    print("Building undirected adjacency...")
    und_adj  = symmetrize_to_undirected(total_comm, n)
    vweights = build_activity_vweights(recv_by_vertex, n)

    # ── METIS initialisation ──────────────────────────────────────────────
    print("METIS initialisation...")
    machine_of = metis_partition(und_adj, args.num_machines,
                                 vweights=vweights, seed=args.seed)
    machine_of = repair_capacity(und_adj, machine_of,
                                 args.nodes_per_machine, args.num_machines)
    init_cut   = compute_cut_weight(und_adj, machine_of)
    print(f"  Initial machine cut weight: {init_cut:,.1f}")

    # ── Build SA edge index ───────────────────────────────────────────────
    print("Building SA edge index...")
    send_edges, recv_edges = build_vertex_edge_index(comm_by_superstep, n)
    sizes = [0] * args.num_machines
    for m in machine_of:
        sizes[m] += 1

    # ── Auto T_start from sampled deltas ─────────────────────────────────
    if args.t_start is None:
        print("Sampling initial temperature...")
        rng_sample = random.Random(args.seed)
        ml_sample  = _initial_machine_loads(
            machine_of, recv_by_vertex, args.num_machines, supersteps, n
        )
        t_start = _sample_initial_temperature(
            machine_of, send_edges, recv_edges, recv_by_vertex,
            ml_sample, supersteps, args.num_machines, sizes,
            args.nodes_per_machine, n, args.c_net, args.c_node, rng_sample
        )
        print(f"  T_start = {t_start:.4f}")
    else:
        t_start = args.t_start

    # ── Simulated annealing ───────────────────────────────────────────────
    print(f"Simulated annealing ({args.sa_iters:,} iterations, "
          f"T {t_start:.4f} → {args.t_end})...")
    machine_of = sa_optimize_machines(
        machine_of, send_edges, recv_edges, recv_by_vertex,
        args.num_machines, args.nodes_per_machine, supersteps, n,
        args.c_net, args.c_node, t_start, args.t_end, args.sa_iters,
        seed=args.seed,
    )
    final_cut = compute_cut_weight(und_adj, machine_of)
    print(f"  Final machine cut weight : {final_cut:,.1f}  "
          f"(Δ = {final_cut - init_cut:+,.1f})")

    # ── Worker assignment ─────────────────────────────────────────────────
    print("Assigning workers within machines (METIS)...")
    worker_of, workers_per_machine = assign_workers(
        und_adj, machine_of, args.num_machines, args.nodes_per_worker,
        seed=args.seed,
    )

    # ── Stats ─────────────────────────────────────────────────────────────
    ss_util  = compute_superstep_utilisation(
        machine_of, recv_by_vertex, args.num_machines, supersteps, n
    )
    avg_util = (sum(ss_util.values()) / (len(supersteps) * args.num_machines)
                if supersteps else 0.0)
    comm_cost, compute_bottleneck = compute_final_cost(
        machine_of, worker_of, comm_by_superstep, recv_by_vertex,
        args.num_machines, supersteps, n,
        args.c_net, args.c_proc, args.c_node,
    )

    # ── Write output ──────────────────────────────────────────────────────
    assignment = {
        str(u): {"machine": int(machine_of[u]), "worker": int(worker_of[u])}
        for u in range(n)
    }
    stats = {
        "num_nodes":                 n,
        "num_machines":              args.num_machines,
        "nodes_per_machine":         args.nodes_per_machine,
        "nodes_per_worker":          args.nodes_per_worker,
        "machine_cut_weight":        final_cut,
        "workers_per_machine":       workers_per_machine,
        "superstep_active_machines": {str(s): v for s, v in ss_util.items()},
        "avg_superstep_utilisation": avg_util,
        "final_comm_cost":           comm_cost,
        "final_compute_bottleneck":  compute_bottleneck,
        "cost_params": {
            "c_net":  args.c_net,
            "c_proc": args.c_proc,
            "c_node": args.c_node,
        },
    }

    payload = {"assignment": assignment, "stats": stats}
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"\nWritten → {args.output}")
    print(f"  machine_cut_weight        : {final_cut:,.1f}")
    print(f"  avg_superstep_utilisation : {avg_util:.3f}")
    print(f"  final_comm_cost           : {comm_cost:,.1f}")
    print(f"  final_compute_bottleneck  : {compute_bottleneck:,.1f}")
    print(f"  superstep active machines:")
    for s in supersteps:
        print(f"    step {s}: {ss_util[s]}/{args.num_machines} machines active")


if __name__ == "__main__":
    main()
