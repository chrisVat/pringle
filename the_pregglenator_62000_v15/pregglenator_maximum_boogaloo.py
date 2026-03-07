#!/usr/bin/env python3

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

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False


# =============================================================================
# 1. Loading traces
# =============================================================================

def load_comm_traces_combiner_aware(traces_dir, max_traces=-1):
    """
    Loads every src_*/merged.csv and treats each src_* file as a separate task.

    Combiner-aware interpretation:
      For communication cost, each unique (task, superstep, src_vertex, dst_vertex)
      contributes PRESENCE, not raw count.
      Raw count is still retained for compute-load estimation.

    Returns
    -------
    out_presence : dict[int, list[(step_key, dst)]]
        For each source vertex u, list of unique outgoing "presence" events.
        Each event means: in this step, u sent something to dst.
    raw_recv_by_vertex : dict[int, dict[step_key, float]]
        Raw total messages received by vertex v in a given step.
    pair_presence : dict[(u, v), float]
        Number of unique step-presence events across all tasks for (u -> v).
        Used for undirected initial METIS graph.
    step_keys : list[tuple[str, int]]
        All distinct (task_name, superstep) keys.
    task_names : list[str]
        Names like src_0, src_1, ...
    """
    paths = sorted(glob.glob(os.path.join(traces_dir, "src_*", "merged.csv")))
    if not paths:
        raise RuntimeError(
            f"No merged.csv files found under {traces_dir}/src_*/\n"
            "Expected layout: <traces_dir>/src_<N>/merged.csv"
        )

    if max_traces > 0:
        paths = paths[:max_traces]
    print(f"  found {len(paths)} trace file(s)")

    out_presence = defaultdict(list)
    raw_recv_by_vertex = defaultdict(lambda: defaultdict(float))
    pair_presence = defaultdict(float)
    step_keys_set = set()
    task_names = []

    for path in tqdm(paths, desc="Reading traces", unit="file"):
        task_name = os.path.basename(os.path.dirname(path))
        task_names.append(task_name)

        if _HAVE_PANDAS:
            df = pd.read_csv(
                path,
                dtype={"superstep": "int32", "src_vertex": "int32",
                       "dst_vertex": "int32", "count": "float32"},
            )
            df = df[(df["src_vertex"] != df["dst_vertex"]) & (df["count"] > 0)]
            grouped = (
                df.groupby(["superstep", "src_vertex", "dst_vertex"], sort=False)["count"]
                .sum()
            )
            rows = grouped.reset_index().itertuples(index=False, name=None)
        else:
            local_counts = defaultdict(float)
            with open(path, newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                ci = {name: i for i, name in enumerate(header)}
                si, ui, vi, ci_ = (ci["superstep"], ci["src_vertex"],
                                    ci["dst_vertex"], ci["count"])
                for row in reader:
                    s, u, v, c = int(row[si]), int(row[ui]), int(row[vi]), float(row[ci_])
                    if u != v and c > 0:
                        local_counts[(s, u, v)] += c
            rows = ((s, u, v, c) for (s, u, v), c in local_counts.items())

        for s, u, v, csum in rows:
            step_key = (task_name, int(s))
            step_keys_set.add(step_key)

            out_presence[int(u)].append((step_key, int(v)))
            raw_recv_by_vertex[int(v)][step_key] += float(csum)
            pair_presence[(int(u), int(v))] += 1.0

    step_keys = sorted(step_keys_set, key=lambda x: (x[0], x[1]))

    return (
        {u: evs for u, evs in out_presence.items()},
        {v: dict(ss) for v, ss in raw_recv_by_vertex.items()},
        dict(pair_presence),
        step_keys,
        task_names,
    )


def infer_num_nodes(pair_presence, explicit_n=None):
    if explicit_n is not None:
        return explicit_n
    mx = -1
    for (u, v) in pair_presence.keys():
        if u > mx:
            mx = u
        if v > mx:
            mx = v
    return mx + 1


# =============================================================================
# 2. Initial graph construction
# =============================================================================

def symmetrize_presence_to_undirected(pair_presence, n):
    """
    Build undirected adjacency using PRESENCE weights, not raw counts.

    weight(u, v) = presence(u -> v) + presence(v -> u)
    """
    directed = defaultdict(dict)
    for (u, v), w in pair_presence.items():
        if 0 <= u < n and 0 <= v < n and u != v and w > 0:
            directed[u][v] = directed[u].get(v, 0.0) + w

    und = [defaultdict(float) for _ in range(n)]
    seen = set()

    for u, nbrs in directed.items():
        for v, w_uv in nbrs.items():
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)

            w_vu = directed.get(v, {}).get(u, 0.0)
            w = w_uv + w_vu
            if w > 0:
                und[u][v] = w
                und[v][u] = w

    return und


def build_activity_vweights(raw_recv_by_vertex, n):
    """
    Simple activity weight for METIS:
      number of distinct steps in which vertex v received anything.
    """
    w = [1] * n
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n:
            w[v] = max(1, len(ss))
    return w


# =============================================================================
# 3. METIS wrappers
# =============================================================================

def _build_pymetis_inputs(und_adj):
    xadj = [0]
    adjncy = []
    eweights = []

    for u in range(len(und_adj)):
        for v, w in sorted(und_adj[u].items()):
            adjncy.append(v)
            eweights.append(max(1, int(round(w))))
        xadj.append(len(adjncy))

    return xadj, adjncy, eweights


def metis_partition(und_adj, nparts, vweights=None):
    if nparts <= 1:
        return [0] * len(und_adj)

    xadj, adjncy, eweights = _build_pymetis_inputs(und_adj)
    adjacency = pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy)

    kwargs = {
        "adjacency": adjacency,
        "eweights": eweights,
    }
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
        pv = part_of[v]
        before = 1 if pv != src else 0
        after = 1 if pv != dst else 0
        delta += w * (after - before)
    return delta


def repair_capacity(und_adj, part_of, capacity, nparts):
    """
    Enforce strict <= capacity nodes per part by moving boundary nodes greedily.
    """
    n = len(part_of)
    sizes = [0] * nparts
    for p in part_of:
        sizes[p] += 1

    overloaded = deque([p for p in range(nparts) if sizes[p] > capacity])
    if not overloaded:
        return part_of

    boundary = _compute_boundary(und_adj, part_of)
    underfull = set([p for p in range(nparts) if sizes[p] < capacity])

    if not underfull:
        raise RuntimeError("Capacity infeasible: all parts full but some overloaded.")

    while overloaded:
        p_over = overloaded.popleft()
        if sizes[p_over] <= capacity:
            continue

        cands = []
        for u in boundary:
            if part_of[u] != p_over:
                continue
            for p_to in underfull:
                delta = _move_delta_cut(und_adj, u, p_over, p_to, part_of)
                cands.append((delta, u, p_to))

        if not cands:
            for u in range(n):
                if part_of[u] != p_over:
                    continue
                for p_to in underfull:
                    delta = _move_delta_cut(und_adj, u, p_over, p_to, part_of)
                    cands.append((delta, u, p_to))

        if not cands:
            raise RuntimeError("Repair failed: no moveable nodes.")

        cands.sort()
        _, u_best, p_to_best = cands[0]

        part_of[u_best] = p_to_best
        sizes[p_over] -= 1
        sizes[p_to_best] += 1

        if sizes[p_to_best] >= capacity:
            underfull.discard(p_to_best)
        if sizes[p_over] > capacity:
            overloaded.append(p_over)

        boundary.add(u_best)
        for nb in und_adj[u_best]:
            boundary.add(nb)

        if not underfull and any(s > capacity for s in sizes):
            raise RuntimeError("Capacity infeasible during repair.")

    return part_of


def induced_subgraph(und_adj, nodes):
    nodes = sorted(nodes)
    old_to_new = {old: i for i, old in enumerate(nodes)}
    k = len(nodes)

    sub = [defaultdict(float) for _ in range(k)]
    for old_u in nodes:
        u = old_to_new[old_u]
        for old_v, w in und_adj[old_u].items():
            if old_v in old_to_new:
                v = old_to_new[old_v]
                if u != v and w > 0:
                    sub[u][v] += w

    return sub, nodes


# =============================================================================
# 4. Sparse state for combiner-aware SA
# =============================================================================

def build_active_recv_raw(raw_recv_by_vertex, n):
    active = {}
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n:
            active[v] = list(ss.items())
    return active


def build_dest_steps(raw_recv_by_vertex, n):
    dest_steps = {}
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n:
            dest_steps[v] = list(ss.keys())
    return dest_steps


def build_sender_part_histograms(part_of, out_presence, n):
    """
    Build:
      sender_hist[(dst, step)][part] = number of source vertices in that part
                                       that sent something to dst in that step
    """
    sender_hist = defaultdict(lambda: defaultdict(int))

    for u, events in out_presence.items():
        if not (0 <= u < n):
            continue
        pu = part_of[u]
        for step_key, dst in events:
            if 0 <= dst < n:
                sender_hist[(dst, step_key)][pu] += 1

    return sender_hist


def _initial_raw_loads(part_of, active_recv_raw, num_parts):
    """
    loads[step_key][part] = raw receive volume hosted by that part in that step
    """
    loads = defaultdict(lambda: [0.0] * num_parts)
    for v, ss_list in active_recv_raw.items():
        p = part_of[v]
        for step_key, rv in ss_list:
            loads[step_key][p] += rv
    return dict(loads)


# =============================================================================
# 5. Combiner-aware machine SA
# =============================================================================

def combiner_comm_delta_for_move(
    v,
    p_from,
    p_to,
    part_of,
    out_presence,
    dest_steps,
    sender_hist,
    c_comm,
):
    """
    Delta in combiner-aware communication objective when vertex v moves
    from p_from to p_to.

    Objective term:
      c_comm * sum_{(dst, step)} number_of_sender_parts_to_dst_excluding_dst_part

    Two ways cost changes:
      1) v as a SOURCE changes which sender-part bucket it contributes to
      2) v as a DESTINATION changes which sender-part bucket is treated as local
    """
    delta = 0.0

    # Source-side change
    for step_key, dst in out_presence.get(v, []):
        pd = part_of[dst]
        hist = sender_hist[(dst, step_key)]

        old_a = hist.get(p_from, 0)
        old_b = hist.get(p_to, 0)

        before = 0
        after = 0

        if p_from != pd:
            before += 1 if old_a > 0 else 0
            after += 1 if (old_a - 1) > 0 else 0

        if p_to != pd:
            before += 1 if old_b > 0 else 0
            after += 1 if (old_b + 1) > 0 else 0

        delta += c_comm * (after - before)

    # Destination-side change
    for step_key in dest_steps.get(v, []):
        hist = sender_hist[(v, step_key)]
        old_local = 1 if hist.get(p_from, 0) > 0 else 0
        new_local = 1 if hist.get(p_to, 0) > 0 else 0

        # cost = total_nonzero_sender_parts - indicator(local_sender_part_present)
        # total_nonzero_sender_parts unchanged here
        delta += c_comm * (old_local - new_local)

    return delta


def apply_sender_hist_move(
    v,
    p_from,
    p_to,
    out_presence,
    sender_hist,
):
    """
    Commit the sender-side histogram updates after accepting a move.
    """
    for step_key, dst in out_presence.get(v, []):
        hist = sender_hist[(dst, step_key)]

        hist[p_from] -= 1
        if hist[p_from] == 0:
            del hist[p_from]

        hist[p_to] += 1


def _sample_temperature_combiner(
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    sender_hist,
    loads,
    num_parts,
    sizes,
    capacity,
    n,
    c_comm,
    c_node,
    rng,
    k=1000,
):
    deltas = []

    for _ in range(k):
        v = rng.randrange(n)
        p_from = part_of[v]
        cands = [p for p in range(num_parts) if p != p_from and sizes[p] < capacity]
        if not cands:
            continue
        p_to = rng.choice(cands)

        d_comm = combiner_comm_delta_for_move(
            v=v,
            p_from=p_from,
            p_to=p_to,
            part_of=part_of,
            out_presence=out_presence,
            dest_steps=dest_steps,
            sender_hist=sender_hist,
            c_comm=c_comm,
        )

        d_compute = 0.0
        for step_key, rv in active_recv_raw.get(v, []):
            sl = loads.get(step_key)
            if sl is None:
                continue
            old_max = max(sl)
            sl[p_from] -= rv
            sl[p_to] += rv
            new_max = max(sl)
            sl[p_from] += rv
            sl[p_to] -= rv
            d_compute += c_node * (new_max - old_max)

        deltas.append(abs(d_comm + d_compute))

    avg = sum(deltas) / len(deltas) if deltas else 1.0
    return max(avg * 2.0, 1e-6)


def sa_optimize_combiner(
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    num_parts,
    capacity,
    n,
    c_comm,
    c_node,
    t_start,
    t_end,
    sa_iters,
    seed=42,
    label="combiner-SA",
):
    rng = random.Random(seed)
    part_of = list(part_of)

    sizes = [0] * num_parts
    for p in part_of:
        sizes[p] += 1

    underfull = set([p for p in range(num_parts) if sizes[p] < capacity])
    loads = _initial_raw_loads(part_of, active_recv_raw, num_parts)
    sender_hist = build_sender_part_histograms(part_of, out_presence, n)

    cooling = 1.0
    if sa_iters > 1 and t_start > t_end:
        cooling = (t_end / t_start) ** (1.0 / (sa_iters - 1))
    T = t_start

    accepted = 0
    post_every = max(1, sa_iters // 200)

    with tqdm(total=sa_iters, desc=label, unit="iter", mininterval=0.5) as pbar:
        for iteration in range(sa_iters):
            v = rng.randrange(n)
            p_from = part_of[v]

            cands = [p for p in underfull if p != p_from]
            if not cands:
                T *= cooling
                pbar.update(1)
                continue

            p_to = rng.choice(cands)

            d_comm = combiner_comm_delta_for_move(
                v=v,
                p_from=p_from,
                p_to=p_to,
                part_of=part_of,
                out_presence=out_presence,
                dest_steps=dest_steps,
                sender_hist=sender_hist,
                c_comm=c_comm,
            )

            d_compute = 0.0
            for step_key, rv in active_recv_raw.get(v, []):
                sl = loads.get(step_key)
                if sl is None:
                    continue
                old_max = max(sl)
                sl[p_from] -= rv
                sl[p_to] += rv
                new_max = max(sl)
                sl[p_from] += rv
                sl[p_to] -= rv
                d_compute += c_node * (new_max - old_max)

            delta = d_comm + d_compute

            if delta < 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                apply_sender_hist_move(
                    v=v,
                    p_from=p_from,
                    p_to=p_to,
                    out_presence=out_presence,
                    sender_hist=sender_hist,
                )

                part_of[v] = p_to
                sizes[p_from] -= 1
                sizes[p_to] += 1

                if sizes[p_from] < capacity:
                    underfull.add(p_from)
                if sizes[p_to] >= capacity:
                    underfull.discard(p_to)

                for step_key, rv in active_recv_raw.get(v, []):
                    sl = loads.get(step_key)
                    if sl is not None:
                        sl[p_from] -= rv
                        sl[p_to] += rv

                accepted += 1

            T *= cooling
            pbar.update(1)

            if (iteration + 1) % post_every == 0:
                pbar.set_postfix(
                    T=f"{T:.2e}",
                    accept=f"{100.0 * accepted / (iteration + 1):.1f}%"
                )

    tqdm.write(
        f"  [{label}] done: accepted {accepted}/{sa_iters} "
        f"({100.0 * accepted / max(sa_iters, 1):.1f}%)"
    )

    return part_of


# =============================================================================
# 6. Worker assignment inside each machine
# =============================================================================

def build_local_out_presence(nodes_m, old_to_local, out_presence):
    local_out = defaultdict(list)
    node_set = set(nodes_m)

    for old_u in nodes_m:
        lu = old_to_local[old_u]
        for step_key, old_dst in out_presence.get(old_u, []):
            if old_dst in node_set:
                local_out[lu].append((step_key, old_to_local[old_dst]))

    return dict(local_out)


def build_local_raw_recv(nodes_m, old_to_local, raw_recv_by_vertex):
    local_recv = {}
    for old_v in nodes_m:
        if old_v in raw_recv_by_vertex:
            local_recv[old_to_local[old_v]] = list(raw_recv_by_vertex[old_v].items())
    return local_recv


def build_local_dest_steps(nodes_m, old_to_local, raw_recv_by_vertex):
    local_dest_steps = {}
    for old_v in nodes_m:
        if old_v in raw_recv_by_vertex:
            local_dest_steps[old_to_local[old_v]] = list(raw_recv_by_vertex[old_v].keys())
    return local_dest_steps


def assign_workers_sa_combiner(
    und_adj,
    machine_of,
    out_presence,
    raw_recv_by_vertex,
    num_machines,
    nodes_per_worker,
    c_proc,
    c_node,
    worker_sa_iters,
    t_end,
    seed,
):
    n = len(und_adj)
    worker_of = [-1] * n
    workers_per_machine = []

    for m in tqdm(range(num_machines), desc="Assigning workers", unit="machine"):
        nodes_m = sorted([u for u in range(n) if machine_of[u] == m])
        if not nodes_m:
            workers_per_machine.append(0)
            continue

        num_workers = max(1, math.ceil(len(nodes_m) / nodes_per_worker))

        if num_workers == 1:
            for u in nodes_m:
                worker_of[u] = 0
            workers_per_machine.append(1)
            continue

        old_to_local = {old: i for i, old in enumerate(nodes_m)}
        sub_adj, _ = induced_subgraph(und_adj, nodes_m)

        local_parts = metis_partition(sub_adj, num_workers)
        local_parts = repair_capacity(sub_adj, local_parts, nodes_per_worker, num_workers)

        local_out_presence = build_local_out_presence(nodes_m, old_to_local, out_presence)
        local_active_recv_raw = build_local_raw_recv(nodes_m, old_to_local, raw_recv_by_vertex)
        local_dest_steps = build_local_dest_steps(nodes_m, old_to_local, raw_recv_by_vertex)

        if worker_sa_iters > 0:
            sizes_w = [0] * num_workers
            for p in local_parts:
                sizes_w[p] += 1

            loads_w = _initial_raw_loads(local_parts, local_active_recv_raw, num_workers)
            sender_hist_w = build_sender_part_histograms(local_parts, local_out_presence, len(nodes_m))

            rng_w = random.Random(seed + 1000 + m)
            t_start_w = _sample_temperature_combiner(
                part_of=local_parts,
                out_presence=local_out_presence,
                dest_steps=local_dest_steps,
                active_recv_raw=local_active_recv_raw,
                sender_hist=sender_hist_w,
                loads=loads_w,
                num_parts=num_workers,
                sizes=sizes_w,
                capacity=nodes_per_worker,
                n=len(nodes_m),
                c_comm=c_proc,
                c_node=c_node,
                rng=rng_w,
                k=min(1000, max(100, len(nodes_m) * 10)),
            )

            print(
                f"  machine {m}: worker SA with {num_workers} workers, "
                f"{len(nodes_m)} nodes, T_start={t_start_w:.4f}"
            )

            local_parts = sa_optimize_combiner(
                part_of=local_parts,
                out_presence=local_out_presence,
                dest_steps=local_dest_steps,
                active_recv_raw=local_active_recv_raw,
                num_parts=num_workers,
                capacity=nodes_per_worker,
                n=len(nodes_m),
                c_comm=c_proc,
                c_node=c_node,
                t_start=t_start_w,
                t_end=t_end,
                sa_iters=worker_sa_iters,
                seed=seed + 2000 + m,
                label=f"worker-SA m={m}",
            )

        for local_u, w in enumerate(local_parts):
            worker_of[nodes_m[local_u]] = w

        workers_per_machine.append(num_workers)

    return worker_of, workers_per_machine


# =============================================================================
# 7. Stats
# =============================================================================

def compute_cut_weight(und_adj, part_of):
    cut = 0.0
    seen = set()
    for u in range(len(und_adj)):
        pu = part_of[u]
        for v, w in und_adj[u].items():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            if part_of[v] != pu:
                cut += w
    return cut


def compute_superstep_utilisation(machine_of, raw_recv_by_vertex, num_machines, step_keys, n):
    util = {}
    for step_key in step_keys:
        active = set()
        for v, ss in raw_recv_by_vertex.items():
            if v < n and step_key in ss:
                active.add(machine_of[v])
        util[step_key] = len(active)
    return util


def compute_combiner_aware_final_comm_cost(
    machine_of,
    worker_of,
    out_presence,
    raw_recv_by_vertex,
    step_keys,
    n,
    c_net,
    c_proc,
):
    """
    Final communication cost with combiner awareness.

    Machine-level cost:
      For each (dst, step), count number of distinct sender machines targeting dst,
      excluding dst's own machine.

    Worker-level same-machine cost:
      For each (dst, step), among same-machine senders only, count distinct sender
      workers targeting dst, excluding dst's own worker.
    """
    machine_hist = defaultdict(lambda: defaultdict(int))
    worker_hist_same_machine = defaultdict(lambda: defaultdict(int))

    for u, events in out_presence.items():
        if not (0 <= u < n):
            continue

        mu = machine_of[u]
        wu = worker_of[u]

        for step_key, dst in events:
            if not (0 <= dst < n):
                continue

            md = machine_of[dst]
            if mu != md:
                machine_hist[(dst, step_key)][mu] += 1
            else:
                worker_hist_same_machine[(dst, step_key)][wu] += 1

    net_cost = 0.0
    for (dst, step_key), hist in machine_hist.items():
        net_cost += c_net * len(hist)

    proc_cost = 0.0
    for (dst, step_key), hist in worker_hist_same_machine.items():
        wd = worker_of[dst]
        proc_cost += c_proc * (len(hist) - (1 if wd in hist else 0))

    return net_cost, proc_cost


def compute_compute_bottleneck(machine_of, raw_recv_by_vertex, num_machines, step_keys, n, c_node):
    total = 0.0
    for step_key in step_keys:
        ml = [0.0] * num_machines
        for v, ss in raw_recv_by_vertex.items():
            if v < n and step_key in ss:
                ml[machine_of[v]] += ss[step_key]
        total += c_node * max(ml)
    return total


def compute_worker_balance(
    worker_of,
    machine_of,
    raw_recv_by_vertex,
    workers_per_machine,
    num_machines,
    step_keys,
    n,
):
    stats = {}

    for m in range(num_machines):
        nw = workers_per_machine[m]
        if nw == 0:
            stats[m] = {
                "sum_max_worker_load": 0.0,
                "sum_avg_worker_load": 0.0,
                "imbalance_ratio": 1.0,
            }
            continue

        sum_max = 0.0
        sum_avg = 0.0

        for step_key in step_keys:
            wl = [0.0] * nw
            for v, ss in raw_recv_by_vertex.items():
                if v < n and machine_of[v] == m and step_key in ss:
                    wl[worker_of[v]] += ss[step_key]

            sum_max += max(wl) if wl else 0.0
            sum_avg += (sum(wl) / nw) if nw > 0 else 0.0

        ratio = (sum_max / sum_avg) if sum_avg > 0 else 1.0
        stats[m] = {
            "sum_max_worker_load": sum_max,
            "sum_avg_worker_load": sum_avg,
            "imbalance_ratio": ratio,
        }

    return stats


# =============================================================================
# 8. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Pregglenator v18: combiner-aware, task-aware, faster two-level SA"
    )
    ap.add_argument("--traces_dir", default="../comm_traces/",
                    help="Directory containing src_*/merged.csv trace files")
    ap.add_argument("--num_nodes", type=int, default=None,
                    help="Graph size N. Inferred if omitted.")
    ap.add_argument("--num_machines", type=int, default=4)
    ap.add_argument("--nodes_per_machine", type=int, default=-1)
    ap.add_argument("--workers_per_machine", type=int, default=4,
                    help="Number of workers per machine. If set, overrides nodes_per_worker.")
    ap.add_argument("--nodes_per_worker", type=int, default=-1)
    
    ap.add_argument("--max_traces", type=int, default=2,
                    help="Maximum number of trace files to load. Loads all if omitted.")

    ap.add_argument("--c_net", type=float, default=100000.0,
                    help="Cost per combined cross-machine delivery")
    ap.add_argument("--c_proc", type=float, default=10.0,
                    help="Cost per combined cross-worker same-machine delivery")
    ap.add_argument("--c_node", type=float, default=1.0,
                    help="Cost weight for compute bottleneck based on raw recv load")

    ap.add_argument("--sa_iters", type=int, default=2_000_000,
                    help="Machine-level SA iterations")
    ap.add_argument("--worker_sa_iters", type=int, default=200_000,
                    help="Worker-level SA iterations per machine")
    ap.add_argument("--t_start", type=float, default=None,
                    help="Machine SA start temperature. Auto-sampled if omitted.")
    ap.add_argument("--t_end", type=float, default=1e-4,
                    help="Shared SA end temperature")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--output", default="pregglenator_v18_combiner_aware.json",
                    help="Output JSON path")

    args = ap.parse_args()

    print("Loading combiner-aware traces...")
    out_presence, raw_recv_by_vertex, pair_presence, step_keys, task_names = load_comm_traces_combiner_aware(args.traces_dir, args.max_traces)

    n = infer_num_nodes(pair_presence, args.num_nodes)
    print(f"  tasks                 : {len(task_names)}")
    print(f"  distinct task-steps   : {len(step_keys)}")
    print(f"  nodes                 : {n:,}")

    if args.nodes_per_machine < 0:
        args.nodes_per_machine = math.ceil(n / args.num_machines)
        print(f"  nodes_per_machine     : {args.nodes_per_machine} (auto)")
    if args.workers_per_machine > 0:
        args.nodes_per_worker = max(1, math.ceil(args.nodes_per_machine / args.workers_per_machine))
        print(f"  workers_per_machine   : {args.workers_per_machine} (explicit)")
        print(f"  nodes_per_worker      : {args.nodes_per_worker} (derived from workers_per_machine)")
    elif args.nodes_per_worker < 0:
        args.nodes_per_worker = max(1, math.ceil(args.nodes_per_machine / 4))
        print(f"  nodes_per_worker      : {args.nodes_per_worker} (auto, ~4 workers/machine)")

    print("Building combiner-aware undirected adjacency for METIS init...")
    und_adj = symmetrize_presence_to_undirected(pair_presence, n)
    vweights = build_activity_vweights(raw_recv_by_vertex, n)

    print("METIS init at machine level...")
    machine_of = metis_partition(und_adj, args.num_machines, vweights=vweights)
    machine_of = repair_capacity(und_adj, machine_of, args.nodes_per_machine, args.num_machines)

    init_cut = compute_cut_weight(und_adj, machine_of)
    print(f"  initial machine cut weight (presence-based): {init_cut:,.1f}")

    active_recv_raw = build_active_recv_raw(raw_recv_by_vertex, n)
    dest_steps = build_dest_steps(raw_recv_by_vertex, n)

    if args.t_start is None:
        print("Sampling initial machine temperature...")
        sizes_s = [0] * args.num_machines
        for p in machine_of:
            sizes_s[p] += 1

        loads_s = _initial_raw_loads(machine_of, active_recv_raw, args.num_machines)
        sender_hist_s = build_sender_part_histograms(machine_of, out_presence, n)
        rng_s = random.Random(args.seed)

        t_start = _sample_temperature_combiner(
            part_of=machine_of,
            out_presence=out_presence,
            dest_steps=dest_steps,
            active_recv_raw=active_recv_raw,
            sender_hist=sender_hist_s,
            loads=loads_s,
            num_parts=args.num_machines,
            sizes=sizes_s,
            capacity=args.nodes_per_machine,
            n=n,
            c_comm=args.c_net,
            c_node=args.c_node,
            rng=rng_s,
            k=min(2000, max(1000, n // 2)),
        )
        print(f"  T_start               : {t_start:.4f}")
    else:
        t_start = args.t_start

    print(
        f"Running machine SA ({args.sa_iters:,} iterations, "
        f"T {t_start:.4f} -> {args.t_end})..."
    )
    machine_of = sa_optimize_combiner(
        part_of=machine_of,
        out_presence=out_presence,
        dest_steps=dest_steps,
        active_recv_raw=active_recv_raw,
        num_parts=args.num_machines,
        capacity=args.nodes_per_machine,
        n=n,
        c_comm=args.c_net,
        c_node=args.c_node,
        t_start=t_start,
        t_end=args.t_end,
        sa_iters=args.sa_iters,
        seed=args.seed,
        label="machine-SA",
    )

    final_cut = compute_cut_weight(und_adj, machine_of)
    print(f"  final machine cut weight (presence-based): {final_cut:,.1f} (delta {final_cut - init_cut:+,.1f})")

    if args.worker_sa_iters > 0:
        print(f"Assigning workers with METIS + combiner-aware SA ({args.worker_sa_iters:,} iters/machine)...")
    else:
        print("Assigning workers with METIS only...")

    worker_of, workers_per_machine = assign_workers_sa_combiner(
        und_adj=und_adj,
        machine_of=machine_of,
        out_presence=out_presence,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        nodes_per_worker=args.nodes_per_worker,
        c_proc=args.c_proc,
        c_node=args.c_node,
        worker_sa_iters=args.worker_sa_iters,
        t_end=args.t_end,
        seed=args.seed,
    )

    print("Computing final stats...")
    ss_util = compute_superstep_utilisation(
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        step_keys=step_keys,
        n=n,
    )
    avg_util = (
        sum(ss_util.values()) / (len(step_keys) * args.num_machines)
        if step_keys else 0.0
    )

    final_net_cost, final_proc_cost = compute_combiner_aware_final_comm_cost(
        machine_of=machine_of,
        worker_of=worker_of,
        out_presence=out_presence,
        raw_recv_by_vertex=raw_recv_by_vertex,
        step_keys=step_keys,
        n=n,
        c_net=args.c_net,
        c_proc=args.c_proc,
    )

    final_compute_cost = compute_compute_bottleneck(
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        step_keys=step_keys,
        n=n,
        c_node=args.c_node,
    )

    worker_balance = compute_worker_balance(
        worker_of=worker_of,
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        workers_per_machine=workers_per_machine,
        num_machines=args.num_machines,
        step_keys=step_keys,
        n=n,
    )

    assignment = {
        str(u): {
            "machine": int(machine_of[u]),
            "worker": int(worker_of[u]),
        }
        for u in range(n)
    }

    stats = {
        "version": "pregglenator_v18_combiner_aware",
        "combiner_model": {
            "machine_level": "one combined transfer per (task_step, dst_vertex, sender_machine)",
            "worker_level": "one combined transfer per (task_step, dst_vertex, sender_worker) within a machine",
            "compute_model": "raw receive volume bottleneck per task_step"
        },
        "num_nodes": n,
        "num_tasks": len(task_names),
        "num_task_steps": len(step_keys),
        "num_machines": args.num_machines,
        "nodes_per_machine": args.nodes_per_machine,
        "nodes_per_worker": args.nodes_per_worker,
        "workers_per_machine": workers_per_machine,
        "machine_cut_weight_presence_based": final_cut,
        "avg_task_step_machine_utilisation": avg_util,
        "final_net_cost_combiner_aware": final_net_cost,
        "final_proc_cost_combiner_aware": final_proc_cost,
        "final_compute_bottleneck_raw": final_compute_cost,
        "total_objective": final_net_cost + final_proc_cost + final_compute_cost,
        "worker_balance": {
            str(m): wb for m, wb in worker_balance.items()
        },
        "cost_params": {
            "c_net": args.c_net,
            "c_proc": args.c_proc,
            "c_node": args.c_node,
        },
    }

    payload = {
        "assignment": assignment,
        "stats": stats,
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"\nWritten -> {args.output}")
    print(f"  machine_cut_weight_presence_based : {final_cut:,.1f}")
    print(f"  avg_task_step_machine_utilisation : {avg_util:.3f}")
    print(f"  final_net_cost_combiner_aware     : {final_net_cost:,.1f}")
    print(f"  final_proc_cost_combiner_aware    : {final_proc_cost:,.1f}")
    print(f"  final_compute_bottleneck_raw      : {final_compute_cost:,.1f}")
    print(f"  total_objective                   : {final_net_cost + final_proc_cost + final_compute_cost:,.1f}")

    print("  worker balance:")
    for m in range(args.num_machines):
        wb = worker_balance[m]
        print(
            f"    machine {m}: ratio={wb['imbalance_ratio']:.3f} "
            f"(sum_max={wb['sum_max_worker_load']:.1f}, "
            f"sum_avg={wb['sum_avg_worker_load']:.1f})"
        )


if __name__ == "__main__":
    main()