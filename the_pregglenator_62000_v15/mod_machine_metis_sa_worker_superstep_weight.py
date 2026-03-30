"""
pregglenator v20 — modulo machine + worker SA + superstep-density edge weights

Changes from v19
----------------
* Edge weight scheme is now hardcoded to superstep-density weighting.
  --edge_weights CLI arg removed.

  und[u][v] = Σ_s  density(s)  for each superstep s where (u,v) is active
  where density(s) = |active edges in s| / |total distinct edges|

  Edges active in dense supersteps (e.g. step 3, 8.5M rows) get higher weight
  than edges only active in sparse supersteps (e.g. step 6, 115 rows).
  Makes training traces necessary and gives METIS/SA varied signal without
  being dominated by raw message volume.

Algorithm
---------
1. Load traces → build superstep-density weighted undirected graph
2. Modulo hash → machine assignment  (u % num_machines)
3. Per machine: METIS init + SA refines worker assignment
4. Report per-machine worker balance and full cost stats

Output JSON
  { "assignment": { "<node_id>": { "machine": int, "worker": int }, ... },
    "stats": { ... } }
"""

import argparse
import glob
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict, deque
 
import pandas as pd
import pymetis
from tqdm import tqdm
 
 
TRACE_CACHE_VERSION = "v20_trace_cache_v1"
GRAPH_CACHE_VERSION = "v20_graph_cache_v1"
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data loading  (pandas + per-file pickle cache)
# ═══════════════════════════════════════════════════════════════════════════════
 
def _load_single_trace(path):
    """
    Load one merged.csv using pandas.
    Returns (comm_ss, recv_vx) dicts for that file.
    Uses a .pkl sidecar cache keyed by mtime.
    """
    cache_path = path + ".v20.pkl"
    mtime = os.path.getmtime(path)
 
    if os.path.isfile(cache_path):
        try:
            t0 = time.perf_counter()
            with open(cache_path, "rb") as fh:
                cached_key, result = pickle.load(fh)
            if cached_key == (TRACE_CACHE_VERSION, mtime):
                print(f"    cache hit  ({time.perf_counter()-t0:.1f}s)  {os.path.basename(os.path.dirname(path))}")
                return result
            print(f"    cache stale for {os.path.basename(os.path.dirname(path))} — rebuilding...")
        except Exception as e:
            print(f"    cache load failed ({e}) — rebuilding...")
 
    t0 = time.perf_counter()
    df = pd.read_csv(
        path,
        dtype={
            "superstep":  "int32",
            "src_vertex": "int32",
            "dst_vertex": "int32",
            "count":      "float32",
        },
    )
    df = df[(df["src_vertex"] != df["dst_vertex"]) & (df["count"] > 0)]
 
    # comm_ss[s][(u,v)] = total count
    comm_ss = defaultdict(lambda: defaultdict(float))
    grouped = df.groupby(["superstep", "src_vertex", "dst_vertex"], sort=False)["count"].sum()
    for (s, u, v), c in grouped.items():
        comm_ss[int(s)][(int(u), int(v))] += float(c)
 
    # recv_vx[v][s] = total recv count
    recv_vx = defaultdict(lambda: defaultdict(float))
    recv_grouped = df.groupby(["superstep", "dst_vertex"], sort=False)["count"].sum()
    for (s, v), c in recv_grouped.items():
        recv_vx[int(v)][int(s)] += float(c)
 
    comm_ss = {s: dict(e) for s, e in comm_ss.items()}
    recv_vx = {v: dict(ss) for v, ss in recv_vx.items()}
 
    result = (comm_ss, recv_vx)
    print(f"    loaded {len(df):,} rows in {time.perf_counter()-t0:.1f}s  "
          f"({os.path.basename(os.path.dirname(path))})")
 
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(((TRACE_CACHE_VERSION, mtime), result), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    cache written → {cache_path}")
    except Exception as e:
        print(f"    cache write failed ({e})")
 
    return result
 
 
def load_comm_traces(traces_dir):
    paths = sorted(glob.glob(os.path.join(traces_dir, "src_*", "merged.csv")))

    if not paths:
        pagerank_path = os.path.join(traces_dir, "pagerank", "merged.csv")
        if os.path.isfile(pagerank_path):
            paths = [pagerank_path]

    if not paths:
        raise RuntimeError(
            f"No merged.csv files found under either:\n"
            f"  {traces_dir}/src_*/merged.csv\n"
            f"  {traces_dir}/pagerank/merged.csv"
        )

    print(f"  {len(paths)} trace file(s): "
          f"{[os.path.basename(os.path.dirname(p)) for p in paths]}")
 
    # Check graph-level cache (covers all traces combined)
    graph_cache_path = os.path.join(traces_dir, ".graph_cache_v20.pkl")
    mtimes = tuple(os.path.getmtime(p) for p in paths)
    graph_cache_key = (GRAPH_CACHE_VERSION, paths, mtimes)
 
    if os.path.isfile(graph_cache_path):
        try:
            t0 = time.perf_counter()
            with open(graph_cache_path, "rb") as fh:
                cached_key, result = pickle.load(fh)
            if cached_key == graph_cache_key:
                print(f"  graph cache hit ({time.perf_counter()-t0:.1f}s)")
                return result
            print("  graph cache stale — rebuilding...")
        except Exception as e:
            print(f"  graph cache load failed ({e}) — rebuilding...")
 
    comm_by_superstep = defaultdict(lambda: defaultdict(float))
    recv_by_vertex    = defaultdict(lambda: defaultdict(float))
 
    for path in tqdm(paths, desc="  Reading traces", unit="file"):
        comm_ss, recv_vx = _load_single_trace(path)
        for s, edges in comm_ss.items():
            for (u, v), c in edges.items():
                comm_by_superstep[s][(u, v)] += c
        for v, ss in recv_vx.items():
            for s, c in ss.items():
                recv_by_vertex[v][s] += c
 
    total_comm = defaultdict(float)
    for edges in comm_by_superstep.values():
        for (u, v), c in edges.items():
            total_comm[(u, v)] += c
 
    result = (
        {s: dict(e)  for s, e  in comm_by_superstep.items()},
        dict(total_comm),
        {v: dict(ss) for v, ss in recv_by_vertex.items()},
    )
 
    try:
        with open(graph_cache_path, "wb") as fh:
            pickle.dump((graph_cache_key, result), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  graph cache written → {graph_cache_path}")
    except Exception as e:
        print(f"  graph cache write failed ({e})")
 
    return result
 
 
def infer_num_nodes(total_comm, explicit_n=None):
    if explicit_n is not None:
        return explicit_n
    mx = max((max(u, v) for u, v in total_comm), default=-1)
    return mx + 1
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Graph construction — superstep-density weights
# ═══════════════════════════════════════════════════════════════════════════════
 
def build_graph_superstep_weights(comm_by_superstep, total_comm, n):
    """
    und[u][v] = Σ_s  density(s)  for each superstep s where (u,v) is active
    where density(s) = |active edges in s| / |total distinct edges|
    """
    total_edges = len(total_comm)
    if total_edges == 0:
        und = [defaultdict(float) for _ in range(n)]
        for (u, v) in total_comm:
            if 0 <= u < n and 0 <= v < n and u != v:
                und[u][v] = 1.0
                und[v][u] = 1.0
        return und
 
    ss_density = {
        s: len(edges) / total_edges
        for s, edges in comm_by_superstep.items()
    }
 
    weighted = defaultdict(float)
    for s, edges in comm_by_superstep.items():
        d = ss_density[s]
        for (u, v) in edges:
            weighted[(u, v)] += d
 
    und  = [defaultdict(float) for _ in range(n)]
    seen = set()
    for (u, v), w in weighted.items():
        if not (0 <= u < n and 0 <= v < n and u != v and w > 0):
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        w_vu     = weighted.get((v, u), 0.0)
        combined = w + w_vu
        und[u][v] = combined
        und[v][u] = combined
 
    return und
 
 
def build_activity_vweights(recv_by_vertex, n):
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
            eweights.append(max(1, int(round(w * 1000))))
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
    return sub, nodes
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Edge index and active-recv helpers
# ═══════════════════════════════════════════════════════════════════════════════
 
def build_vertex_edge_index(comm_by_superstep, n):
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
    return {v: list(ss.items()) for v, ss in recv_by_vertex.items() if v < n}
 
 
def build_intra_machine_edge_index(nodes_m, old_to_local, send_edges, recv_edges):
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
    result = {}
    for old_v in nodes_m:
        if old_v in recv_by_vertex:
            result[old_to_local[old_v]] = list(recv_by_vertex[old_v].items())
    return result
 
 
def _initial_loads(part_of, active_recv, num_parts):
    loads = defaultdict(lambda: [0.0] * num_parts)
    for v, ss_list in active_recv.items():
        p = part_of[v]
        for s, rv in ss_list:
            loads[s][p] += rv
    return dict(loads)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Generic SA optimizer
# ═══════════════════════════════════════════════════════════════════════════════
 
def _sample_temperature(
    part_of, send_edges, recv_edges, active_recv,
    loads, num_parts, sizes, capacity,
    n, c_comm, c_node, rng, k=1000,
):
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
            sl[p_from] += rv
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
    rng      = random.Random(seed)
    part_of  = list(part_of)
    sizes    = [0] * num_parts
    for p in part_of:
        sizes[p] += 1
 
    loads     = _initial_loads(part_of, active_recv, num_parts)
    underfull = set(p for p in range(num_parts) if sizes[p] < capacity)
 
    cooling    = ((t_end / t_start) ** (1.0 / (sa_iters - 1))
                  if sa_iters > 1 and t_start > t_end else 1.0)
    T          = t_start
    accepted   = 0
    post_every = max(1, sa_iters // 200)
 
    with tqdm(total=sa_iters, desc=label, unit="iter", mininterval=0.5) as pbar:
        for iteration in range(sa_iters):
            v      = rng.randrange(n)
            p_from = part_of[v]
 
            cands = [p for p in underfull if p != p_from]
            if not cands:
                T *= cooling
                pbar.update(1)
                continue
 
            p_to = rng.choice(cands)
 
            d_comm = 0.0
            for dst, c in send_edges.get(v, []):
                pd      = part_of[dst]
                d_comm += c_comm * c * ((0 if pd == p_to else 1) -
                                        (0 if pd == p_from else 1))
            for src, c in recv_edges.get(v, []):
                ps      = part_of[src]
                d_comm += c_comm * c * ((0 if ps == p_to else 1) -
                                        (0 if ps == p_from else 1))
 
            d_compute = 0.0
            for s, rv in active_recv.get(v, []):
                sl = loads.get(s)
                if sl is None:
                    continue
                old_max    = max(sl)
                sl[p_from] -= rv
                sl[p_to]   += rv
                new_max    = max(sl)
                sl[p_from] += rv
                sl[p_to]   -= rv
                d_compute  += c_node * (new_max - old_max)
 
            delta = d_comm + d_compute
 
            if delta < 0.0 or rng.random() < math.exp(-delta / T):
                part_of[v]    = p_to
                sizes[p_from] -= 1
                sizes[p_to]   += 1
 
                if sizes[p_from] < capacity:
                    underfull.add(p_from)
                if sizes[p_to] >= capacity:
                    underfull.discard(p_to)
 
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
 
        sub_adj, _ = induced_subgraph(und_adj, nodes_m)
        local_parts = metis_partition(sub_adj, num_workers, seed=seed + m + 1)
        local_parts = repair_capacity(sub_adj, local_parts, nodes_per_worker, num_workers)
 
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
        description="Pregglenator v20_fast — pandas loading + pickle cache"
    )
    ap.add_argument("--traces_dir",        default='../comm_traces/',
                    help="Directory containing src_*/merged.csv trace files")
    ap.add_argument("--num_nodes",         type=int,   default=None)
    ap.add_argument("--num_machines",      type=int,   default=15)
    ap.add_argument("--nodes_per_machine", type=int,   default=-1)
    ap.add_argument("--nodes_per_worker",  type=int,   default=-1)
    ap.add_argument("--c_net",             type=float, default=500.0,
                    help="Cost weight for cross-machine messages — stats reporting only")
    ap.add_argument("--c_proc",            type=float, default=1.0,
                    help="Cost weight for cross-worker intra-machine messages")
    ap.add_argument("--c_node",            type=float, default=1.0,
                    help="Cost weight per unit of vertex compute load")
    ap.add_argument("--worker_sa_iters",   type=int,   default=500_000,
                    help="Worker-level SA iterations per machine. 0 = METIS only.")
    ap.add_argument("--t_end",             type=float, default=1e-4)
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--output",            default='partition_v20_fast.json')
    ap.add_argument("--clear_cache",       action="store_true",
                    help="Delete all .v20.pkl and graph cache files before running")
    args = ap.parse_args()
 
    # Optionally wipe caches
    if args.clear_cache:
        paths = glob.glob(os.path.join(args.traces_dir, "src_*", "merged.csv.v20.pkl"))
        graph_cache = os.path.join(args.traces_dir, ".graph_cache_v20.pkl")
        for p in paths + ([graph_cache] if os.path.isfile(graph_cache) else []):
            os.remove(p)
            print(f"  removed cache: {p}")
 
    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading comm traces...")
    t_load = time.perf_counter()
    comm_by_superstep, total_comm, recv_by_vertex = load_comm_traces(args.traces_dir)
    print(f"  load total: {time.perf_counter()-t_load:.2f}s")
 
    supersteps = sorted(comm_by_superstep.keys())
    n_edges    = sum(len(e) for e in comm_by_superstep.values())
    print(f"  {len(supersteps)} supersteps  |  {n_edges:,} (superstep, edge) pairs")
 
    n = infer_num_nodes(total_comm, args.num_nodes)
    print(f"  {n:,} nodes (IDs 0–{n-1})")
 
    if args.nodes_per_machine < 0:
        args.nodes_per_machine = math.ceil(n / args.num_machines)
        print(f"  nodes_per_machine = {args.nodes_per_machine} (auto)")
    if args.nodes_per_worker < 0:
        args.nodes_per_worker = math.ceil(args.nodes_per_machine / 4)
        print(f"  nodes_per_worker  = {args.nodes_per_worker} (auto, ~4 workers/machine)")
 
    # ── Build superstep-density weighted undirected graph ────────────────────
    print("Building superstep-density weighted undirected adjacency...")
    und_adj = build_graph_superstep_weights(comm_by_superstep, total_comm, n)
 
    total_edges = len(total_comm)
    print("  Superstep densities:")
    for s in supersteps:
        d = len(comm_by_superstep[s]) / total_edges
        print(f"    step {s}: {len(comm_by_superstep[s]):>9,} active edges  "
              f"density={d:.4f}")
 
    # ── Modulo machine assignment ─────────────────────────────────────────────
    print("Machine assignment: modulo hash (u % num_machines)...")
    machine_of    = [u % args.num_machines for u in range(n)]
    machine_sizes = [machine_of.count(m) for m in range(args.num_machines)]
    print(f"  Machine sizes: {machine_sizes}")
    machine_cut   = compute_cut_weight(und_adj, machine_of)
    print(f"  Machine cut weight (informational): {machine_cut:,.4f}")
 
    # ── Build SA edge index ───────────────────────────────────────────────────
    print("Building SA edge index...")
    send_edges, recv_edges = build_vertex_edge_index(comm_by_superstep, n)
    active_recv            = build_active_recv(recv_by_vertex, n)
 
    # ── Worker assignment ─────────────────────────────────────────────────────
    if args.worker_sa_iters > 0:
        print(f"Assigning workers (METIS + SA, {args.worker_sa_iters:,} iters/machine)...")
    else:
        print("Assigning workers (METIS only)...")
 
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
        "machine_assignment":        "modulo_hash",
        "edge_weights":              "superstep_density",
        "machine_cut_weight":        machine_cut,
        "machine_sizes":             machine_sizes,
        "workers_per_machine":       workers_per_machine,
        "superstep_active_machines": {str(s): v for s, v in ss_util.items()},
        "avg_superstep_utilisation": avg_util,
        "final_comm_cost":           comm_cost,
        "final_compute_bottleneck":  compute_bottleneck,
        "worker_balance":            {str(m): wb for m, wb in worker_balance.items()},
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
    print(f"  machine_assignment        : modulo hash")
    print(f"  edge_weights              : superstep_density")
    print(f"  machine_cut_weight        : {machine_cut:,.4f}")
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