"""
pregglenator v17 — two-level SA: machine + worker aware

Changes from v16 (pregglenator_electric_boogaloo2.py)
------------------------------------------------------
Performance
  * send/recv edge index now stores (neighbor, total_count) pairs aggregated
    across all supersteps instead of (superstep, neighbor, count).  This
    shrinks each vertex's edge list from O(supersteps × degree) to O(degree),
    making the SA comm-delta inner loop proportionally faster.
  * active_recv[v] = [(s, rv), ...] for only the supersteps where v is active;
    replaces the previous full-superstep scan every SA iteration.
  * underfull set replaces per-iteration list comprehension over all partitions.
  * randrange(n) replaces rng.choice(vertices_list) for vertex selection.
  * Generic sa_optimize() shared by machine and worker levels.

Worker awareness
  * Worker assignment = METIS initialisation + SA refinement within each machine.
  * Worker SA optimises:
        c_proc × cross-worker intra-machine traffic
      + c_node × Σ_s  max_worker_in_machine(recv-load in s)
  * New CLI arg: --worker_sa_iters  (iterations per machine, default 500 000).
    Set to 0 to skip worker SA and fall back to pure-METIS worker assignment.
  * Worker-level load-balance statistics added to output.

Algorithm
---------
1. Load traces → undirected graph          [same as v16]
2. METIS + repair → machine assignment     [same as v16]
3. SA refines machine assignment           [same cost model, faster impl]
4. Per machine: METIS init + SA refines worker assignment  [NEW]
5. Report full three-component cost + per-machine worker balance  [extended]

Output JSON (same schema as v16)
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
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

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

    comm_ss = defaultdict(lambda: defaultdict(float))
    recv_vx = defaultdict(lambda: defaultdict(float))

    for path in tqdm(paths, desc="  Reading traces", unit="file"):
        with open(path, newline="") as fh:
            for row in tqdm(csv.DictReader(fh), desc=f"    {os.path.basename(os.path.dirname(path))}",
                            unit="row", leave=False):
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Graph construction
# ═══════════════════════════════════════════════════════════════════════════════

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
    w = [1] * n
    for v, ss in recv_by_vertex.items():
        if v < n:
            w[v] = max(1, len(ss))
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  METIS wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pymetis_inputs(und_adj):
    xadj     = [0]
    adjncy   = []
    eweights = []
    for u in range(len(und_adj)):
        for v, w in sorted(und_adj[u].items()):
            adjncy.append(v)
            eweights.append(max(1, int(round(w))))
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights


def metis_partition(und_adj, nparts, vweights=None, seed=42):
    if nparts <= 1:
        return [0] * len(und_adj)
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
        pu = part_of[u]
        for v in und_adj[u]:
            if part_of[v] != pu:
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
        part_of[u_best]   = p_to_best
        sizes[p_over]    -= 1
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


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Edge index and active-recv helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_vertex_edge_index(comm_by_superstep, n):
    """
    Build global send/recv indices aggregated across all supersteps.

    Returns
    -------
    send_edges : {u: [(v, total_count), ...]}
    recv_edges : {v: [(u, total_count), ...]}

    Storing total counts (not per-superstep) shrinks each list from
    O(supersteps × degree) to O(degree), making the SA comm-delta loop faster.
    Superstep information for load tracking is handled separately via active_recv.
    """
    send_agg = defaultdict(lambda: defaultdict(float))
    recv_agg = defaultdict(lambda: defaultdict(float))
    for edges in tqdm(comm_by_superstep.values(), desc="  Edge index", unit="superstep"):
        for (u, v), c in edges.items():
            if 0 <= u < n and 0 <= v < n:
                send_agg[u][v] += c
                recv_agg[v][u] += c
    return (
        {u: list(nbrs.items()) for u, nbrs in send_agg.items()},
        {v: list(nbrs.items()) for v, nbrs in recv_agg.items()},
    )


def build_active_recv(recv_by_vertex, n):
    """
    active_recv[v] = [(s, rv), ...] for supersteps where v receives > 0 messages.
    Only iterate active supersteps in the SA hot loop instead of all supersteps.
    """
    return {v: list(ss.items()) for v, ss in recv_by_vertex.items() if v < n}


def build_intra_machine_edge_index(nodes_m, old_to_local, send_edges, recv_edges):
    """
    Build send/recv indices for intra-machine edges only, using local IDs.

    Filters the already-aggregated global send_edges/recv_edges rather than
    re-scanning comm_by_superstep.  Cost is O(Σ degree(v) for v in nodes_m)
    total across all machines, versus the previous O(M × Σ|E_s|).
    """
    local_send = {}
    local_recv = {}
    for old_u in nodes_m:
        lu = old_to_local[old_u]
        intra_send = [(old_to_local[v], c)
                      for v, c in send_edges.get(old_u, [])
                      if v in old_to_local]
        if intra_send:
            local_send[lu] = intra_send
        intra_recv = [(old_to_local[u], c)
                      for u, c in recv_edges.get(old_u, [])
                      if u in old_to_local]
        if intra_recv:
            local_recv[lu] = intra_recv
    return local_send, local_recv


def build_active_recv_local(recv_by_vertex, nodes_m, old_to_local):
    """active_recv in local IDs for nodes belonging to one machine."""
    result = {}
    for old_v in nodes_m:
        if old_v in recv_by_vertex:
            result[old_to_local[old_v]] = list(recv_by_vertex[old_v].items())
    return result


def _initial_loads(part_of, active_recv, num_parts):
    """
    Initialise per-superstep partition loads.
    loads[s][p] = total recv messages on partition p in superstep s.
    """
    loads = defaultdict(lambda: [0.0] * num_parts)
    for v, ss_list in active_recv.items():
        p = part_of[v]
        for s, rv in ss_list:
            loads[s][p] += rv
    return dict(loads)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Generic SA optimizer  (machine level and worker level share this)
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_temperature(
    part_of, send_edges, recv_edges, active_recv,
    loads, num_parts, sizes, capacity,
    n, c_comm, c_node, rng, k=1000,
):
    """Estimate T_start as 2× mean |Δcost| over k random moves."""
    deltas = []
    for _ in range(k):
        v      = rng.randrange(n)
        p_from = part_of[v]
        cands  = [p for p in range(num_parts)
                  if p != p_from and sizes[p] < capacity]
        if not cands:
            continue
        p_to = rng.choice(cands)

        d = 0.0
        for dst, c in send_edges.get(v, []):
            pd  = part_of[dst]
            d  += c_comm * c * ((0 if pd == p_to else 1) - (0 if pd == p_from else 1))
        for src, c in recv_edges.get(v, []):
            ps  = part_of[src]
            d  += c_comm * c * ((0 if ps == p_to else 1) - (0 if ps == p_from else 1))

        for s, rv in active_recv.get(v, []):
            sl = loads.get(s)
            if sl is None:
                continue
            old_max    = max(sl)
            sl[p_from] -= rv
            sl[p_to]   += rv
            new_max    = max(sl)
            sl[p_from] += rv   # restore
            sl[p_to]   -= rv
            d += c_node * (new_max - old_max)

        deltas.append(abs(d))

    avg = sum(deltas) / len(deltas) if deltas else 1.0
    return max(avg * 2.0, 1e-6)


def sa_optimize(
    part_of, send_edges, recv_edges, active_recv,
    num_parts, capacity, n,
    c_comm, c_node,
    t_start, t_end, sa_iters,
    seed=42, label="SA",
):
    """
    Generic SA partition refiner — used at both machine and worker level.

    Minimises:
      c_comm × Σ cross-partition message counts (aggregated across supersteps)
    + c_node × Σ_s  max_part( recv-load on partition in superstep s )

    Capacity constraint: ≤ capacity nodes per partition (strictly enforced by
    only proposing moves to non-saturated partitions).

    Parameters
    ----------
    send_edges  : {v: [(neighbor, total_count), ...]}   (agg. across supersteps)
    recv_edges  : {v: [(neighbor, total_count), ...]}
    active_recv : {v: [(s, recv_load), ...]}            (per-superstep loads)
    """
    rng      = random.Random(seed)
    part_of  = list(part_of)
    sizes    = [0] * num_parts
    for p in part_of:
        sizes[p] += 1

    loads     = _initial_loads(part_of, active_recv, num_parts)
    underfull = set(p for p in range(num_parts) if sizes[p] < capacity)

    cooling  = ((t_end / t_start) ** (1.0 / (sa_iters - 1))
                if sa_iters > 1 and t_start > t_end else 1.0)
    T        = t_start
    accepted = 0
    # Update postfix every 0.5% of iterations to keep overhead low
    post_every = max(1, sa_iters // 200)

    with tqdm(total=sa_iters, desc=label, unit="iter", mininterval=0.5) as pbar:
        for iteration in range(sa_iters):
            v      = rng.randrange(n)
            p_from = part_of[v]

            # Candidate partitions: underfull and not the current one.
            # underfull is a maintained set so no O(num_parts) scan per iteration.
            cands = [p for p in underfull if p != p_from]
            if not cands:
                T *= cooling
                pbar.update(1)
                continue

            p_to = rng.choice(cands)

            # ── Δ communication cost ─────────────────────────────────────────
            # Edges are (neighbor, total_count) aggregated across all supersteps.
            d_comm = 0.0
            for dst, c in send_edges.get(v, []):
                pd      = part_of[dst]
                d_comm += c_comm * c * ((0 if pd == p_to else 1) -
                                        (0 if pd == p_from else 1))
            for src, c in recv_edges.get(v, []):
                ps      = part_of[src]
                d_comm += c_comm * c * ((0 if ps == p_to else 1) -
                                        (0 if ps == p_from else 1))

            # ── Δ compute-bottleneck cost ────────────────────────────────────
            # Only iterate supersteps where v is actually active (active_recv).
            d_compute = 0.0
            for s, rv in active_recv.get(v, []):
                sl = loads.get(s)
                if sl is None:
                    continue
                old_max    = max(sl)
                sl[p_from] -= rv
                sl[p_to]   += rv
                new_max    = max(sl)
                sl[p_from] += rv   # restore
                sl[p_to]   -= rv
                d_compute  += c_node * (new_max - old_max)

            delta = d_comm + d_compute

            # ── Accept / reject ──────────────────────────────────────────────
            if delta < 0.0 or rng.random() < math.exp(-delta / T):
                part_of[v]    = p_to
                sizes[p_from] -= 1
                sizes[p_to]   += 1

                # Maintain underfull set
                if sizes[p_from] < capacity:
                    underfull.add(p_from)
                if sizes[p_to] >= capacity:
                    underfull.discard(p_to)

                # Commit load changes
                for s, rv in active_recv.get(v, []):
                    sl = loads.get(s)
                    if sl is not None and rv:
                        sl[p_from] -= rv
                        sl[p_to]   += rv
                accepted += 1

            T *= cooling
            pbar.update(1)

            if (iteration + 1) % post_every == 0:
                pbar.set_postfix(T=f"{T:.2e}",
                                 accept=f"{100*accepted/(iteration+1):.1f}%")

    tqdm.write(f"  [{label}] done — accepted {accepted}/{sa_iters} "
               f"({100*accepted/sa_iters:.1f}%)")
    return part_of


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Worker assignment: METIS init + per-machine worker SA
# ═══════════════════════════════════════════════════════════════════════════════

def assign_workers_sa(
    und_adj, machine_of, send_edges, recv_edges, recv_by_vertex,
    num_machines, nodes_per_worker,
    c_proc, c_node,
    worker_sa_iters, t_end, seed,
):
    """
    Assign workers within each machine using METIS initialisation followed by
    SA refinement.

    The worker SA optimises:
        c_proc × cross-worker intra-machine traffic
      + c_node × Σ_s  max_worker_in_machine( recv-load in superstep s )

    Parameters
    ----------
    worker_sa_iters : int
        SA iterations per machine. Set to 0 to skip SA and use METIS only.

    Returns
    -------
    worker_of           : list[int]  global node → local worker ID
    workers_per_machine : list[int]  number of workers per machine
    """
    n         = len(und_adj)
    worker_of = [-1] * n
    wpm       = []

    for m in tqdm(range(num_machines), desc="  Machines", unit="machine"):
        nodes_m     = sorted(u for u in range(n) if machine_of[u] == m)
        num_workers = max(1, math.ceil(len(nodes_m) / nodes_per_worker))

        if not nodes_m:
            wpm.append(0)
            continue

        if num_workers == 1:
            for u in nodes_m:
                worker_of[u] = 0
            wpm.append(1)
            continue

        old_to_local = {old: i for i, old in enumerate(nodes_m)}
        nlocal       = len(nodes_m)

        # ── METIS initialisation ─────────────────────────────────────────────
        sub_adj, _ = induced_subgraph(und_adj, nodes_m)
        local_parts = metis_partition(sub_adj, num_workers, seed=seed + m + 1)
        local_parts = repair_capacity(sub_adj, local_parts, nodes_per_worker, num_workers)

        # ── Worker SA refinement ─────────────────────────────────────────────
        if worker_sa_iters > 0:
            local_send, local_recv = build_intra_machine_edge_index(
                nodes_m, old_to_local, send_edges, recv_edges
            )
            local_active_recv = build_active_recv_local(
                recv_by_vertex, nodes_m, old_to_local
            )

            sizes_w = [0] * num_workers
            for p in local_parts:
                sizes_w[p] += 1
            loads_s = _initial_loads(local_parts, local_active_recv, num_workers)

            rng_s     = random.Random(seed + m)
            t_start_w = _sample_temperature(
                local_parts, local_send, local_recv, local_active_recv,
                loads_s, num_workers, sizes_w, nodes_per_worker,
                nlocal, c_proc, c_node, rng_s,
            )
            print(f"  [machine {m}] worker SA: {num_workers} workers, "
                  f"{nlocal} nodes, T_start={t_start_w:.4f}")

            local_parts = sa_optimize(
                local_parts, local_send, local_recv, local_active_recv,
                num_workers, nodes_per_worker, nlocal,
                c_proc, c_node,
                t_start_w, t_end, worker_sa_iters,
                seed=seed + m + 1000, label=f"worker-SA m={m}",
            )

        for local_u, w in enumerate(local_parts):
            worker_of[nodes_m[local_u]] = w

        wpm.append(num_workers)

    return worker_of, wpm


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Stats
# ═══════════════════════════════════════════════════════════════════════════════

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
    num_machines, supersteps, n, c_net, c_proc, c_node,
):
    """Full three-component cost after worker assignment is known."""
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


def compute_worker_balance(
    worker_of, machine_of, recv_by_vertex,
    workers_per_machine, num_machines, supersteps, n,
):
    """
    Per-machine worker load-balance statistics.

    For each machine returns:
      sum_max_worker_load  — Σ_s max_worker(recv-load in s)   (the bottleneck term)
      sum_avg_worker_load  — Σ_s avg_worker(recv-load in s)
      imbalance_ratio      — sum_max / sum_avg  (1.0 = perfect balance)
    """
    stats = {}
    for m in range(num_machines):
        nw = workers_per_machine[m]
        if nw == 0:
            stats[m] = {"sum_max_worker_load": 0.0,
                        "sum_avg_worker_load": 0.0,
                        "imbalance_ratio":     1.0}
            continue

        sum_max = 0.0
        sum_avg = 0.0
        for s in supersteps:
            wl = [0.0] * nw
            for v, ss in recv_by_vertex.items():
                if s in ss and v < n and machine_of[v] == m:
                    wl[worker_of[v]] += ss[s]
            sum_max += max(wl)
            sum_avg += sum(wl) / nw

        ratio = sum_max / sum_avg if sum_avg > 0 else 1.0
        stats[m] = {
            "sum_max_worker_load": sum_max,
            "sum_avg_worker_load": sum_avg,
            "imbalance_ratio":     ratio,
        }
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Superstep-aware Pregel partitioner — two-level SA (v17)"
    )
    ap.add_argument("--traces_dir", default='../comm_traces/',
                    help="Directory containing src_*/merged.csv trace files")
    ap.add_argument("--num_nodes",         type=int,   default=None,
                    help="Graph size N (inferred from data if omitted)")
    ap.add_argument("--num_machines",      type=int, default=4)
    ap.add_argument("--nodes_per_machine", type=int,  default=-1)
    ap.add_argument("--nodes_per_worker",  type=int,   default=-1)
    ap.add_argument("--c_net",             type=float, default=1000000.0,
                    help="Cost weight for cross-machine messages (default 1000000)")
    ap.add_argument("--c_proc",            type=float, default=10.0,
                    help="Cost weight for cross-worker same-machine messages (default 3)")
    ap.add_argument("--c_node",            type=float, default=1.0,
                    help="Cost weight per unit of vertex compute load (default 1)")
    ap.add_argument("--sa_iters",          type=int,   default=5_000_000,
                    help="Machine-level SA iterations (default 5 000 000)")
    ap.add_argument("--worker_sa_iters",   type=int,   default=500_000,
                    help="Worker-level SA iterations per machine (default 500 000). "
                         "Set to 0 to use pure-METIS worker assignment.")
    ap.add_argument("--t_start",           type=float, default=None,
                    help="Machine SA start temperature (auto-sampled if omitted)")
    ap.add_argument("--t_end",             type=float, default=1e-4,
                    help="SA end temperature, shared by machine and worker SA (default 1e-4)")
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--output",            default='theboogalo3__1_10_1000000.json',
                    help="Output JSON path")
    args = ap.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading comm traces...")
    comm_by_superstep, total_comm, recv_by_vertex = load_comm_traces(args.traces_dir)
    supersteps = sorted(comm_by_superstep.keys())
    n_edges    = sum(len(e) for e in comm_by_superstep.values())
    print(f"  {len(supersteps)} supersteps  |  {n_edges:,} (superstep, edge) pairs")

    n = infer_num_nodes(total_comm, args.num_nodes)
    print(f"  {n:,} nodes (IDs 0–{n-1})")

    if args.nodes_per_machine < 0:
        args.nodes_per_machine = math.ceil(n / args.num_machines)
        print(f"  nodes_per_machine = {args.nodes_per_machine} (auto: ceil({n}/{args.num_machines}))")
    if args.nodes_per_worker < 0:
        args.nodes_per_worker = math.ceil(args.nodes_per_machine / 4)
        print(f"  nodes_per_worker  = {args.nodes_per_worker} (auto: ceil({args.nodes_per_machine}/4), ~4 workers/machine)")

    # ── Build undirected graph ────────────────────────────────────────────────
    print("Building undirected adjacency...")
    und_adj  = symmetrize_to_undirected(total_comm, n)
    vweights = build_activity_vweights(recv_by_vertex, n)

    # ── METIS initialisation ──────────────────────────────────────────────────
    print("METIS initialisation (machine level)...")
    machine_of = metis_partition(und_adj, args.num_machines,
                                 vweights=vweights, seed=args.seed)
    machine_of = repair_capacity(und_adj, machine_of,
                                 args.nodes_per_machine, args.num_machines)
    init_cut   = compute_cut_weight(und_adj, machine_of)
    print(f"  Initial machine cut weight: {init_cut:,.1f}")

    # ── Build SA edge index (aggregated, O(degree) per vertex) ───────────────
    print("Building SA edge index...")
    send_edges, recv_edges = build_vertex_edge_index(comm_by_superstep, n)
    active_recv            = build_active_recv(recv_by_vertex, n)

    # ── Auto T_start from sampled deltas ─────────────────────────────────────
    if args.t_start is None:
        print("Sampling initial machine temperature...")
        rng_s   = random.Random(args.seed)
        sizes_s = [0] * args.num_machines
        for m in machine_of:
            sizes_s[m] += 1
        loads_s = _initial_loads(machine_of, active_recv, args.num_machines)
        t_start = _sample_temperature(
            machine_of, send_edges, recv_edges, active_recv,
            loads_s, args.num_machines, sizes_s, args.nodes_per_machine,
            n, args.c_net, args.c_node, rng_s,
        )
        print(f"  T_start = {t_start:.4f}")
    else:
        t_start = args.t_start

    # ── Machine-level SA ──────────────────────────────────────────────────────
    print(f"Machine SA ({args.sa_iters:,} iterations, "
          f"T {t_start:.4f} → {args.t_end})...")
    machine_of = sa_optimize(
        machine_of, send_edges, recv_edges, active_recv,
        args.num_machines, args.nodes_per_machine, n,
        args.c_net, args.c_node,
        t_start, args.t_end, args.sa_iters,
        seed=args.seed, label="machine-SA",
    )
    final_cut = compute_cut_weight(und_adj, machine_of)
    print(f"  Final machine cut weight : {final_cut:,.1f}  "
          f"(Δ = {final_cut - init_cut:+,.1f})")

    # ── Worker assignment: METIS init + worker SA ─────────────────────────────
    if args.worker_sa_iters > 0:
        print(f"Assigning workers within machines (METIS + SA, "
              f"{args.worker_sa_iters:,} iters/machine)...")
    else:
        print("Assigning workers within machines (METIS only)...")
    worker_of, workers_per_machine = assign_workers_sa(
        und_adj, machine_of, send_edges, recv_edges, recv_by_vertex,
        args.num_machines, args.nodes_per_worker,
        args.c_proc, args.c_node,
        args.worker_sa_iters, args.t_end, args.seed,
    )

    # ── Stats ─────────────────────────────────────────────────────────────────
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
    worker_balance = compute_worker_balance(
        worker_of, machine_of, recv_by_vertex,
        workers_per_machine, args.num_machines, supersteps, n,
    )

    # ── Write output ──────────────────────────────────────────────────────────
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
        "worker_balance": {
            str(m): wb for m, wb in worker_balance.items()
        },
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
    print(f"  worker balance (imbalance_ratio = max/avg load):")
    for m in range(args.num_machines):
        wb = worker_balance[m]
        print(f"    machine {m}: ratio={wb['imbalance_ratio']:.3f}  "
              f"(sum_max={wb['sum_max_worker_load']:.1f}, "
              f"sum_avg={wb['sum_avg_worker_load']:.1f})")
    print(f"  superstep active machines:")
    for s in supersteps:
        print(f"    step {s}: {ss_util[s]}/{args.num_machines} machines active")


if __name__ == "__main__":
    main()
