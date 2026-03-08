#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import math
import os
import pickle
import random
import time
import heapq
from collections import defaultdict, deque

from tqdm import tqdm

try:
    import pymetis
    _HAVE_PYMETIS = True
except ImportError:
    _HAVE_PYMETIS = False

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False


TRACE_CACHE_VERSION = "opt_trace_cache_v1"


# =============================================================================
# 1. Small helpers
# =============================================================================

def _pick_col(fieldnames, candidates, required=True):
    lower_map = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise RuntimeError(
            f"Could not find required CSV column. Need one of {candidates}, "
            f"but got {fieldnames}"
        )
    return None


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _ensure_undirected_unique(adj):
    out = []
    for nbrs in adj:
        out.append(sorted(set(nbrs)) if nbrs else [])
    return out


# =============================================================================
# 2. Loading traces with OLD per-trace cache behavior
# =============================================================================

def _load_single_trace_combiner_aware(path):
    """
    Load one src_*/merged.csv and return per-trace data.

    Returns
    -------
    task_name : str
    out_presence_local : dict[int, list[(step_key, dst)]]
    raw_recv_local : dict[int, dict[step_key, float]]
    pair_presence_local : dict[(u, v), float]
    step_keys_local : list[tuple[str, int]]
    cache_status : str
        one of {"hit", "miss", "stale", "failed"}
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"Trace file not found: {path}")

    task_name = os.path.basename(os.path.dirname(path))
    cache_path = path + ".opt.pkl"
    trace_mtime = os.path.getmtime(path)

    if os.path.isfile(cache_path):
        try:
            t_load = time.perf_counter()
            with open(cache_path, "rb") as fh:
                cached_key, result = pickle.load(fh)
            if cached_key == (TRACE_CACHE_VERSION, trace_mtime):
                print(
                    f"    cache hit for {task_name} "
                    f"({time.perf_counter() - t_load:.1f}s)"
                )
                return (*result, "hit")
            else:
                print(f"    cache stale for {task_name} - rebuilding...")
                cache_status = "stale"
        except Exception as e:
            print(f"    cache load failed for {task_name} ({e}) - rebuilding...")
            cache_status = "failed"
    else:
        print(f"    cache miss for {task_name} - building...")
        cache_status = "miss"

    out_presence_local = defaultdict(list)
    raw_recv_local = defaultdict(lambda: defaultdict(float))
    pair_presence_local = defaultdict(float)
    step_keys_set = set()

    if _HAVE_PANDAS:
        df = pd.read_csv(path)

        src_col = _pick_col(
            list(df.columns),
            ["src_vertex", "src", "src_id", "source", "from", "source_vertex"],
        )
        dst_col = _pick_col(
            list(df.columns),
            ["dst_vertex", "dst", "dst_id", "dest", "destination", "to", "dest_vertex"],
        )
        step_col = _pick_col(
            list(df.columns),
            ["superstep", "step", "ss", "iteration"],
        )
        cnt_col = _pick_col(
            list(df.columns),
            ["count", "cnt", "num_messages", "messages", "msg_count", "weight"],
            required=False,
        )

        df[src_col] = pd.to_numeric(df[src_col], errors="coerce").fillna(-1).astype("int64")
        df[dst_col] = pd.to_numeric(df[dst_col], errors="coerce").fillna(-1).astype("int64")
        df[step_col] = pd.to_numeric(df[step_col], errors="coerce").fillna(0).astype("int64")
        if cnt_col is None:
            df["_count_tmp"] = 1.0
            cnt_col_use = "_count_tmp"
        else:
            df[cnt_col] = pd.to_numeric(df[cnt_col], errors="coerce").fillna(0.0).astype("float64")
            cnt_col_use = cnt_col

        df = df[
            (df[src_col] >= 0) &
            (df[dst_col] >= 0) &
            (df[src_col] != df[dst_col]) &
            (df[cnt_col_use] > 0)
        ]

        grouped = df.groupby(
            [step_col, src_col, dst_col],
            sort=False
        )[cnt_col_use].sum()

        print(f"    pandas: {len(df):,} rows -> {len(grouped):,} unique (s,u,v) pairs")

        recv_grouped = df.groupby(
            [step_col, dst_col],
            sort=False
        )[cnt_col_use].sum()

        unique_steps = df[step_col].unique()
        step_key_of = {int(s): (task_name, int(s)) for s in unique_steps}
        step_keys_set = set(step_key_of.values())

        for (s, v), csum in tqdm(
            recv_grouped.items(),
            total=len(recv_grouped),
            desc=f"    recv index {task_name}",
            unit="pair",
            mininterval=0.5,
            leave=False,
        ):
            raw_recv_local[int(v)][step_key_of[int(s)]] += float(csum)

        presence_df = grouped.reset_index()[[step_col, src_col, dst_col]]

        for s, u, v in tqdm(
            presence_df.itertuples(index=False, name=None),
            total=len(presence_df),
            desc=f"    presence index {task_name}",
            unit="edge",
            mininterval=0.5,
            leave=False,
        ):
            step_key = step_key_of[int(s)]
            u = int(u)
            v = int(v)
            out_presence_local[u].append((step_key, v))
            pair_presence_local[(u, v)] += 1.0

    else:
        local_counts = defaultdict(float)

        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise RuntimeError(f"No CSV header found in: {path}")

            src_col = _pick_col(
                reader.fieldnames,
                ["src_vertex", "src", "src_id", "source", "from", "source_vertex"],
            )
            dst_col = _pick_col(
                reader.fieldnames,
                ["dst_vertex", "dst", "dst_id", "dest", "destination", "to", "dest_vertex"],
            )
            step_col = _pick_col(
                reader.fieldnames,
                ["superstep", "step", "ss", "iteration"],
            )
            cnt_col = _pick_col(
                reader.fieldnames,
                ["count", "cnt", "num_messages", "messages", "msg_count", "weight"],
                required=False,
            )

            for row in tqdm(
                reader,
                desc=f"    reading {task_name}",
                unit="row",
                mininterval=0.5,
                leave=False,
            ):
                s = _safe_int(row.get(step_col, 0), default=0)
                u = _safe_int(row.get(src_col, -1), default=-1)
                v = _safe_int(row.get(dst_col, -1), default=-1)
                c = 1.0 if cnt_col is None else _safe_float(row.get(cnt_col, 0.0), default=0.0)

                if u >= 0 and v >= 0 and u != v and c > 0:
                    local_counts[(s, u, v)] += c

        unique_steps = sorted({s for (s, _, _) in local_counts.keys()})
        step_key_of = {int(s): (task_name, int(s)) for s in unique_steps}
        step_keys_set = set(step_key_of.values())

        for (s, u, v), csum in tqdm(
            local_counts.items(),
            total=len(local_counts),
            desc=f"    index {task_name}",
            unit="pair",
            mininterval=0.5,
            leave=False,
        ):
            step_key = step_key_of[int(s)]
            out_presence_local[int(u)].append((step_key, int(v)))
            raw_recv_local[int(v)][step_key] += float(csum)
            pair_presence_local[(int(u), int(v))] += 1.0

    step_keys_local = sorted(step_keys_set, key=lambda x: (x[0], x[1]))

    result = (
        task_name,
        dict(out_presence_local),
        {v: dict(ss) for v, ss in raw_recv_local.items()},
        dict(pair_presence_local),
        step_keys_local,
    )

    try:
        t_write = time.perf_counter()
        with open(cache_path, "wb") as fh:
            pickle.dump(
                ((TRACE_CACHE_VERSION, trace_mtime), result),
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"    cache written for {task_name} ({time.perf_counter() - t_write:.1f}s)")
    except Exception as e:
        print(f"    cache write failed for {task_name} ({e}) - continuing without cache")

    return (*result, cache_status)


def load_comm_traces_combiner_aware(traces_dir, max_traces=-1, trace_paths=None):
    """
    Loads every src_*/merged.csv and treats each src_* file as a separate task.

    Combiner-aware interpretation:
      For communication cost, each unique (task, superstep, src_vertex, dst_vertex)
      contributes PRESENCE, not raw count.
      Raw count is still retained for compute-load estimation.
    """
    if trace_paths:
        paths = list(trace_paths)
    else:
        paths = sorted(glob.glob(os.path.join(traces_dir, "src_*", "merged.csv")))
        if max_traces > 0:
            paths = paths[:max_traces]

    if not paths:
        raise RuntimeError(
            f"No merged.csv files found.\n"
            f"Expected either --trace_paths or {traces_dir}/src_*/merged.csv"
        )

    print(f"  found {len(paths)} trace file(s)")
    print(f"  trace cache version   : {TRACE_CACHE_VERSION}")

    out_presence = defaultdict(list)
    raw_recv_by_vertex = defaultdict(lambda: defaultdict(float))
    pair_presence = defaultdict(float)
    step_keys_set = set()
    task_names = []

    num_hits = 0
    num_misses = 0
    num_stale = 0
    num_failed = 0

    for path in tqdm(paths, desc="Reading traces", unit="file"):
        task_name, out_local, recv_local, pair_local, step_keys_local, cache_status = _load_single_trace_combiner_aware(path)
        task_names.append(task_name)

        if cache_status == "hit":
            num_hits += 1
        elif cache_status == "miss":
            num_misses += 1
        elif cache_status == "stale":
            num_stale += 1
        else:
            num_failed += 1

        for u, evs in out_local.items():
            out_presence[u].extend(evs)

        for v, ss in recv_local.items():
            dst_map = raw_recv_by_vertex[v]
            for step_key, csum in ss.items():
                dst_map[step_key] += csum

        for key, w in pair_local.items():
            pair_presence[key] += w

        for step_key in step_keys_local:
            step_keys_set.add(step_key)

    step_keys = sorted(step_keys_set, key=lambda x: (x[0], x[1]))

    print(
        f"  cache summary         : "
        f"{num_hits} hit, {num_misses} miss, {num_stale} stale, {num_failed} failed"
    )

    return (
        dict(out_presence),
        {v: dict(ss) for v, ss in raw_recv_by_vertex.items()},
        dict(pair_presence),
        step_keys,
        task_names,
    )


def infer_num_nodes(pair_presence, raw_recv_by_vertex, out_presence, explicit_n=None):
    if explicit_n is not None:
        return explicit_n

    mx = -1

    for (u, v) in pair_presence.keys():
        if u > mx:
            mx = u
        if v > mx:
            mx = v

    for v in raw_recv_by_vertex.keys():
        if v > mx:
            mx = v

    for u, evs in out_presence.items():
        if u > mx:
            mx = u
        for _, dst in evs:
            if dst > mx:
                mx = dst

    return mx + 1


# =============================================================================
# 3. Initial graph construction
# =============================================================================

def symmetrize_presence_to_undirected(pair_presence, n):
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
    w = [1] * n
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n:
            w[v] = max(1, len(ss))
    return w


def greedy_balanced_partition(und_adj, num_parts, capacity, seed=42):
    rng = random.Random(seed)
    n = len(und_adj)

    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=lambda u: len(und_adj[u]), reverse=True)

    part_of = [-1] * n
    sizes = [0] * num_parts

    for u in order:
        nbr_part_score = defaultdict(float)
        for v, w in und_adj[u].items():
            pv = part_of[v]
            if pv >= 0:
                nbr_part_score[pv] += w

        candidates = [p for p in range(num_parts) if sizes[p] < capacity]
        if not candidates:
            raise RuntimeError("Capacity infeasible in greedy_balanced_partition")

        best_key = None
        best_parts = []

        for p in candidates:
            key = (nbr_part_score.get(p, 0.0), -sizes[p], -p)
            if best_key is None or key > best_key:
                best_key = key
                best_parts = [p]
            elif key == best_key:
                best_parts.append(p)

        chosen = rng.choice(best_parts)
        part_of[u] = chosen
        sizes[chosen] += 1

    return part_of


# =============================================================================
# 4. METIS wrappers
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
    if not _HAVE_PYMETIS:
        raise RuntimeError("pymetis is not installed")

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


def metis_or_fallback(und_adj, nparts, capacity, vweights=None, seed=42, label="metis"):
    if not _HAVE_PYMETIS:
        tqdm.write(f"  [{label}] pymetis missing, falling back to greedy init")
        return greedy_balanced_partition(und_adj, nparts, capacity, seed=seed)

    try:
        t0 = time.perf_counter()
        parts = metis_partition(und_adj, nparts, vweights=vweights)
        dt = time.perf_counter() - t0
        tqdm.write(f"  [{label}] METIS finished in {dt:.2f}s")
        return parts
    except Exception as e:
        tqdm.write(f"  [{label}] METIS failed ({e}), falling back to greedy init")
        return greedy_balanced_partition(und_adj, nparts, capacity, seed=seed)


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
# 5. Sparse state
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
    loads = defaultdict(lambda: [0.0] * num_parts)
    for v, ss_list in active_recv_raw.items():
        p = part_of[v]
        for step_key, rv in ss_list:
            loads[step_key][p] += rv
    return dict(loads)


def build_step_vertices(out_presence, dest_steps, n):
    by_step = defaultdict(set)

    for u, events in out_presence.items():
        if 0 <= u < n:
            for step_key, _dst in events:
                by_step[step_key].add(u)

    for v, steps in dest_steps.items():
        if 0 <= v < n:
            for step_key in steps:
                by_step[step_key].add(v)

    return {k: list(vs) for k, vs in by_step.items()}


def build_machine_net_state(part_of, sender_hist, step_keys, num_parts):
    net_step = {sk: [0.0] * num_parts for sk in step_keys}

    for (dst, step_key), hist in sender_hist.items():
        md = part_of[dst]
        contrib = len(hist) - (1 if md in hist else 0)
        net_step[step_key][md] += contrib

    return net_step


def build_worker_mem_state(part_of, sender_hist, step_keys, num_parts):
    mem_step = {sk: [0.0] * num_parts for sk in step_keys}

    for (dst, step_key), hist in sender_hist.items():
        wd = part_of[dst]
        contrib = len(hist) - (1 if wd in hist else 0)
        mem_step[step_key][wd] += contrib

    return mem_step


def _touched_steps_for_vertex(v, out_presence, dest_steps, active_recv_raw):
    touched = set()
    for step_key, _dst in out_presence.get(v, []):
        touched.add(step_key)
    for step_key in dest_steps.get(v, []):
        touched.add(step_key)
    for step_key, _rv in active_recv_raw.get(v, []):
        touched.add(step_key)
    return touched


def step_max_cost(load_vec, comm_vec, c_node, c_comm):
    best = -1.0
    best_part = 0
    for p in range(len(load_vec)):
        val = c_node * load_vec[p] + c_comm * comm_vec[p]
        if val > best:
            best = val
            best_part = p
    return best, best_part


class MaxStepState:
    def __init__(self, step_keys, loads, comm_step, c_node, c_comm):
        self.step_keys = list(step_keys)
        self.loads = loads
        self.comm_step = comm_step
        self.c_node = c_node
        self.c_comm = c_comm

        self.best_val = {}
        self.best_part = {}
        self.heap = []

        for sk in self.step_keys:
            val, part = step_max_cost(
                self.loads.get(sk, []),
                self.comm_step.get(sk, []),
                self.c_node,
                self.c_comm,
            )
            self.best_val[sk] = val
            self.best_part[sk] = part
            heapq.heappush(self.heap, (-val, sk, part))

    def recompute_step(self, step_key):
        val, part = step_max_cost(
            self.loads.get(step_key, []),
            self.comm_step.get(step_key, []),
            self.c_node,
            self.c_comm,
        )
        self.best_val[step_key] = val
        self.best_part[step_key] = part
        heapq.heappush(self.heap, (-val, step_key, part))

    def recompute_steps(self, step_keys):
        for sk in step_keys:
            self.recompute_step(sk)

    def get_worst(self):
        while self.heap:
            neg_val, sk, part = self.heap[0]
            cur_val = self.best_val.get(sk, None)
            cur_part = self.best_part.get(sk, None)
            if cur_val is not None and abs(-neg_val - cur_val) < 1e-12 and part == cur_part:
                return sk, part, cur_val
            heapq.heappop(self.heap)
        return None, 0, 0.0


def choose_targeted_move(
    part_of,
    sizes,
    capacity,
    step_vertices,
    worst_step,
    worst_part,
    loads,
    comm_step,
    rng,
    top_k_parts=3,
):
    if worst_step is None:
        return None, None

    candidates = [v for v in step_vertices.get(worst_step, []) if part_of[v] == worst_part]
    if not candidates:
        return None, None

    v = rng.choice(candidates)

    underfull = [p for p in range(len(sizes)) if p != worst_part and sizes[p] < capacity]
    if not underfull:
        return None, None

    load_vec = loads.get(worst_step, [0.0] * len(sizes))
    comm_vec = comm_step.get(worst_step, [0.0] * len(sizes))
    scored = [(load_vec[p] + comm_vec[p], p) for p in underfull]
    scored.sort()

    dest_pool = [p for _, p in scored[:max(1, min(top_k_parts, len(scored)))]]
    p_to = rng.choice(dest_pool)
    return v, p_to


# =============================================================================
# 6. Machine-level SA for heaviest-machine objective
# =============================================================================

def machine_move_delta_max(
    v,
    p_from,
    p_to,
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    sender_hist,
    loads,
    net_step,
    c_net,
    c_node,
):
    touched_steps = _touched_steps_for_vertex(v, out_presence, dest_steps, active_recv_raw)
    if not touched_steps:
        return 0.0, touched_steps

    any_step = next(iter(net_step)) if net_step else None
    num_parts = len(net_step[any_step]) if any_step is not None else 0

    temp_loads = {}
    temp_net = {}

    for step_key in touched_steps:
        temp_loads[step_key] = list(loads.get(step_key, [0.0] * num_parts))
        temp_net[step_key] = list(net_step.get(step_key, [0.0] * num_parts))

    for step_key, dst in out_presence.get(v, []):
        hist = sender_hist[(dst, step_key)]
        md = part_of[dst]

        old_a = hist.get(p_from, 0)
        old_b = hist.get(p_to, 0)
        old_contrib = len(hist) - (1 if md in hist else 0)

        new_nonzero = len(hist)
        if old_a == 1:
            new_nonzero -= 1
        if old_b == 0:
            new_nonzero += 1

        new_local = 1 if (
            (md == p_from and old_a - 1 > 0) or
            (md == p_to and old_b + 1 > 0) or
            (md != p_from and md != p_to and hist.get(md, 0) > 0)
        ) else 0

        new_contrib = new_nonzero - new_local
        temp_net[step_key][md] += (new_contrib - old_contrib)

    for step_key in dest_steps.get(v, []):
        hist = sender_hist[(v, step_key)]
        old_contrib = len(hist) - (1 if p_from in hist else 0)
        new_contrib = len(hist) - (1 if p_to in hist else 0)
        temp_net[step_key][p_from] -= old_contrib
        temp_net[step_key][p_to] += new_contrib

    for step_key, rv in active_recv_raw.get(v, []):
        temp_loads[step_key][p_from] -= rv
        temp_loads[step_key][p_to] += rv

    delta = 0.0
    for step_key in touched_steps:
        old_val, _ = step_max_cost(
            loads.get(step_key, temp_loads[step_key]),
            net_step.get(step_key, temp_net[step_key]),
            c_node,
            c_net,
        )
        new_val, _ = step_max_cost(temp_loads[step_key], temp_net[step_key], c_node, c_net)
        delta += (new_val - old_val)

    return delta, touched_steps


def apply_machine_move_max(
    v,
    p_from,
    p_to,
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    sender_hist,
    loads,
    net_step,
):
    for step_key, dst in out_presence.get(v, []):
        hist = sender_hist[(dst, step_key)]
        md = part_of[dst]

        old_contrib = len(hist) - (1 if md in hist else 0)

        hist[p_from] -= 1
        if hist[p_from] == 0:
            del hist[p_from]
        hist[p_to] += 1

        new_contrib = len(hist) - (1 if md in hist else 0)
        net_step[step_key][md] += (new_contrib - old_contrib)

    for step_key in dest_steps.get(v, []):
        hist = sender_hist[(v, step_key)]
        old_contrib = len(hist) - (1 if p_from in hist else 0)
        new_contrib = len(hist) - (1 if p_to in hist else 0)

        net_step[step_key][p_from] -= old_contrib
        net_step[step_key][p_to] += new_contrib

    for step_key, rv in active_recv_raw.get(v, []):
        loads[step_key][p_from] -= rv
        loads[step_key][p_to] += rv

    part_of[v] = p_to


def sample_temperature_machine_max(
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    sender_hist,
    loads,
    net_step,
    step_vertices,
    capacity,
    c_net,
    c_node,
    rng,
    k=400,
):
    num_parts = max(part_of) + 1 if part_of else 0
    sizes = [0] * num_parts
    for p in part_of:
        sizes[p] += 1

    step_state = MaxStepState(list(net_step.keys()), loads, net_step, c_node, c_net)
    deltas = []

    for _ in range(k):
        worst_step, worst_part, _ = step_state.get_worst()
        v, p_to = choose_targeted_move(
            part_of=part_of,
            sizes=sizes,
            capacity=capacity,
            step_vertices=step_vertices,
            worst_step=worst_step,
            worst_part=worst_part,
            loads=loads,
            comm_step=net_step,
            rng=rng,
        )
        if v is None:
            continue

        p_from = part_of[v]
        d, _ = machine_move_delta_max(
            v=v,
            p_from=p_from,
            p_to=p_to,
            part_of=part_of,
            out_presence=out_presence,
            dest_steps=dest_steps,
            active_recv_raw=active_recv_raw,
            sender_hist=sender_hist,
            loads=loads,
            net_step=net_step,
            c_net=c_net,
            c_node=c_node,
        )
        deltas.append(abs(d))

    avg = sum(deltas) / len(deltas) if deltas else 1.0
    return max(2.0 * avg, 1e-6)


def sa_optimize_machine_max_targeted(
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    num_parts,
    capacity,
    n,
    c_net,
    c_node,
    t_start,
    t_end,
    sa_iters,
    seed=42,
    label="machine-SA",
):
    rng = random.Random(seed)
    part_of = list(part_of)

    sizes = [0] * num_parts
    for p in part_of:
        sizes[p] += 1

    sender_hist = build_sender_part_histograms(part_of, out_presence, n)
    loads = _initial_raw_loads(part_of, active_recv_raw, num_parts)
    step_keys = sorted(set(loads.keys()) | {sk for evs in out_presence.values() for sk, _ in evs}, key=lambda x: (x[0], x[1]))
    net_step = build_machine_net_state(part_of, sender_hist, step_keys, num_parts)
    step_vertices = build_step_vertices(out_presence, dest_steps, n)
    step_state = MaxStepState(step_keys, loads, net_step, c_node, c_net)

    cooling = 1.0
    if sa_iters > 1 and t_start > t_end:
        cooling = (t_end / t_start) ** (1.0 / (sa_iters - 1))
    T = t_start

    accepted = 0
    post_every = max(1, sa_iters // 200)

    with tqdm(total=sa_iters, desc=label, unit="iter", mininterval=0.5) as pbar:
        for it in range(sa_iters):
            worst_step, worst_part, worst_val = step_state.get_worst()

            if rng.random() < 0.88:
                v, p_to = choose_targeted_move(
                    part_of=part_of,
                    sizes=sizes,
                    capacity=capacity,
                    step_vertices=step_vertices,
                    worst_step=worst_step,
                    worst_part=worst_part,
                    loads=loads,
                    comm_step=net_step,
                    rng=rng,
                )
            else:
                v = rng.randrange(n)
                p_from = part_of[v]
                underfull = [p for p in range(num_parts) if p != p_from and sizes[p] < capacity]
                p_to = rng.choice(underfull) if underfull else None

            if v is None or p_to is None:
                T *= cooling
                pbar.update(1)
                continue

            p_from = part_of[v]
            if p_from == p_to:
                T *= cooling
                pbar.update(1)
                continue

            delta, touched_steps = machine_move_delta_max(
                v=v,
                p_from=p_from,
                p_to=p_to,
                part_of=part_of,
                out_presence=out_presence,
                dest_steps=dest_steps,
                active_recv_raw=active_recv_raw,
                sender_hist=sender_hist,
                loads=loads,
                net_step=net_step,
                c_net=c_net,
                c_node=c_node,
            )

            if delta < 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
                apply_machine_move_max(
                    v=v,
                    p_from=p_from,
                    p_to=p_to,
                    part_of=part_of,
                    out_presence=out_presence,
                    dest_steps=dest_steps,
                    active_recv_raw=active_recv_raw,
                    sender_hist=sender_hist,
                    loads=loads,
                    net_step=net_step,
                )
                sizes[p_from] -= 1
                sizes[p_to] += 1
                step_state.recompute_steps(touched_steps)
                accepted += 1

            T *= cooling
            pbar.update(1)

            if (it + 1) % post_every == 0:
                pbar.set_postfix(
                    T=f"{T:.2e}",
                    worst=f"{worst_val:.3e}",
                    accept=f"{100.0 * accepted / (it + 1):.1f}%"
                )

    tqdm.write(
        f"  [{label}] done: accepted {accepted}/{sa_iters} "
        f"({100.0 * accepted / max(sa_iters, 1):.1f}%)"
    )

    return part_of


# =============================================================================
# 7. Worker assignment inside each machine
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


def worker_move_delta_max(
    v,
    p_from,
    p_to,
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    sender_hist,
    loads,
    mem_step,
    c_mem,
    c_node,
):
    touched_steps = _touched_steps_for_vertex(v, out_presence, dest_steps, active_recv_raw)
    if not touched_steps:
        return 0.0, touched_steps

    any_step = next(iter(mem_step)) if mem_step else None
    num_parts = len(mem_step[any_step]) if any_step is not None else 0

    temp_loads = {}
    temp_mem = {}

    for step_key in touched_steps:
        temp_loads[step_key] = list(loads.get(step_key, [0.0] * num_parts))
        temp_mem[step_key] = list(mem_step.get(step_key, [0.0] * num_parts))

    for step_key, dst in out_presence.get(v, []):
        hist = sender_hist[(dst, step_key)]
        wd = part_of[dst]

        old_a = hist.get(p_from, 0)
        old_b = hist.get(p_to, 0)
        old_contrib = len(hist) - (1 if wd in hist else 0)

        new_nonzero = len(hist)
        if old_a == 1:
            new_nonzero -= 1
        if old_b == 0:
            new_nonzero += 1

        new_local = 1 if (
            (wd == p_from and old_a - 1 > 0) or
            (wd == p_to and old_b + 1 > 0) or
            (wd != p_from and wd != p_to and hist.get(wd, 0) > 0)
        ) else 0

        new_contrib = new_nonzero - new_local
        temp_mem[step_key][wd] += (new_contrib - old_contrib)

    for step_key in dest_steps.get(v, []):
        hist = sender_hist[(v, step_key)]
        old_contrib = len(hist) - (1 if p_from in hist else 0)
        new_contrib = len(hist) - (1 if p_to in hist else 0)
        temp_mem[step_key][p_from] -= old_contrib
        temp_mem[step_key][p_to] += new_contrib

    for step_key, rv in active_recv_raw.get(v, []):
        temp_loads[step_key][p_from] -= rv
        temp_loads[step_key][p_to] += rv

    delta = 0.0
    for step_key in touched_steps:
        old_val, _ = step_max_cost(
            loads.get(step_key, temp_loads[step_key]),
            mem_step.get(step_key, temp_mem[step_key]),
            c_node,
            c_mem,
        )
        new_val, _ = step_max_cost(temp_loads[step_key], temp_mem[step_key], c_node, c_mem)
        delta += (new_val - old_val)

    return delta, touched_steps


def apply_worker_move_max(
    v,
    p_from,
    p_to,
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    sender_hist,
    loads,
    mem_step,
):
    for step_key, dst in out_presence.get(v, []):
        hist = sender_hist[(dst, step_key)]
        wd = part_of[dst]

        old_contrib = len(hist) - (1 if wd in hist else 0)

        hist[p_from] -= 1
        if hist[p_from] == 0:
            del hist[p_from]
        hist[p_to] += 1

        new_contrib = len(hist) - (1 if wd in hist else 0)
        mem_step[step_key][wd] += (new_contrib - old_contrib)

    for step_key in dest_steps.get(v, []):
        hist = sender_hist[(v, step_key)]
        old_contrib = len(hist) - (1 if p_from in hist else 0)
        new_contrib = len(hist) - (1 if p_to in hist else 0)

        mem_step[step_key][p_from] -= old_contrib
        mem_step[step_key][p_to] += new_contrib

    for step_key, rv in active_recv_raw.get(v, []):
        loads[step_key][p_from] -= rv
        loads[step_key][p_to] += rv

    part_of[v] = p_to


def sample_temperature_worker_max(
    part_of,
    out_presence,
    dest_steps,
    active_recv_raw,
    sender_hist,
    loads,
    mem_step,
    step_vertices,
    capacity,
    c_mem,
    c_node,
    rng,
    k=400,
):
    num_parts = max(part_of) + 1 if part_of else 0
    sizes = [0] * num_parts
    for p in part_of:
        sizes[p] += 1

    step_state = MaxStepState(list(mem_step.keys()), loads, mem_step, c_node, c_mem)
    deltas = []

    for _ in range(k):
        worst_step, worst_part, _ = step_state.get_worst()
        v, p_to = choose_targeted_move(
            part_of=part_of,
            sizes=sizes,
            capacity=capacity,
            step_vertices=step_vertices,
            worst_step=worst_step,
            worst_part=worst_part,
            loads=loads,
            comm_step=mem_step,
            rng=rng,
        )
        if v is None:
            continue

        p_from = part_of[v]
        d, _ = worker_move_delta_max(
            v=v,
            p_from=p_from,
            p_to=p_to,
            part_of=part_of,
            out_presence=out_presence,
            dest_steps=dest_steps,
            active_recv_raw=active_recv_raw,
            sender_hist=sender_hist,
            loads=loads,
            mem_step=mem_step,
            c_mem=c_mem,
            c_node=c_node,
        )
        deltas.append(abs(d))

    avg = sum(deltas) / len(deltas) if deltas else 1.0
    return max(2.0 * avg, 1e-6)


def assign_workers_sa_combiner_max_targeted(
    und_adj,
    machine_of,
    out_presence,
    raw_recv_by_vertex,
    num_machines,
    nodes_per_worker,
    c_mem,
    c_node,
    worker_sa_iters,
    t_end,
    seed,
    worker_init="greedy",
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

        if worker_init == "metis":
            local_parts = metis_or_fallback(
                sub_adj,
                num_workers,
                nodes_per_worker,
                vweights=None,
                seed=seed + 1000 + m,
                label=f"worker-init m={m}",
            )
        else:
            local_parts = greedy_balanced_partition(
                sub_adj,
                num_workers,
                nodes_per_worker,
                seed=seed + 1000 + m,
            )

        local_parts = repair_capacity(sub_adj, local_parts, nodes_per_worker, num_workers)

        local_out_presence = build_local_out_presence(nodes_m, old_to_local, out_presence)
        local_active_recv_raw = build_local_raw_recv(nodes_m, old_to_local, raw_recv_by_vertex)
        local_dest_steps = build_local_dest_steps(nodes_m, old_to_local, raw_recv_by_vertex)

        if worker_sa_iters > 0:
            sender_hist = build_sender_part_histograms(local_parts, local_out_presence, len(nodes_m))
            loads = _initial_raw_loads(local_parts, local_active_recv_raw, num_workers)
            step_keys = sorted(set(loads.keys()) | {sk for evs in local_out_presence.values() for sk, _ in evs}, key=lambda x: (x[0], x[1]))
            mem_step = build_worker_mem_state(local_parts, sender_hist, step_keys, num_workers)
            step_vertices = build_step_vertices(local_out_presence, local_dest_steps, len(nodes_m))
            rng_w = random.Random(seed + 1000 + m)

            t_start_w = sample_temperature_worker_max(
                part_of=local_parts,
                out_presence=local_out_presence,
                dest_steps=local_dest_steps,
                active_recv_raw=local_active_recv_raw,
                sender_hist=sender_hist,
                loads=loads,
                mem_step=mem_step,
                step_vertices=step_vertices,
                capacity=nodes_per_worker,
                c_mem=c_mem,
                c_node=c_node,
                rng=rng_w,
                k=min(400, max(100, len(nodes_m) * 4)),
            )

            print(
                f"  machine {m}: worker SA with {num_workers} workers, "
                f"{len(nodes_m)} nodes, T_start={t_start_w:.4f}"
            )

            cooling = 1.0
            if worker_sa_iters > 1 and t_start_w > t_end:
                cooling = (t_end / t_start_w) ** (1.0 / (worker_sa_iters - 1))
            T = t_start_w

            sizes = [0] * num_workers
            for p in local_parts:
                sizes[p] += 1

            accepted = 0
            post_every = max(1, worker_sa_iters // 200)
            step_state = MaxStepState(step_keys, loads, mem_step, c_node, c_mem)

            with tqdm(total=worker_sa_iters, desc=f"worker-SA m={m}", unit="iter", mininterval=0.5) as pbar:
                for it in range(worker_sa_iters):
                    worst_step, worst_worker, worst_val = step_state.get_worst()

                    if rng_w.random() < 0.88:
                        v, p_to = choose_targeted_move(
                            part_of=local_parts,
                            sizes=sizes,
                            capacity=nodes_per_worker,
                            step_vertices=step_vertices,
                            worst_step=worst_step,
                            worst_part=worst_worker,
                            loads=loads,
                            comm_step=mem_step,
                            rng=rng_w,
                        )
                    else:
                        v = rng_w.randrange(len(nodes_m))
                        p_from = local_parts[v]
                        underfull = [p for p in range(num_workers) if p != p_from and sizes[p] < nodes_per_worker]
                        p_to = rng_w.choice(underfull) if underfull else None

                    if v is None or p_to is None:
                        T *= cooling
                        pbar.update(1)
                        continue

                    p_from = local_parts[v]
                    if p_from == p_to:
                        T *= cooling
                        pbar.update(1)
                        continue

                    delta, touched_steps = worker_move_delta_max(
                        v=v,
                        p_from=p_from,
                        p_to=p_to,
                        part_of=local_parts,
                        out_presence=local_out_presence,
                        dest_steps=local_dest_steps,
                        active_recv_raw=local_active_recv_raw,
                        sender_hist=sender_hist,
                        loads=loads,
                        mem_step=mem_step,
                        c_mem=c_mem,
                        c_node=c_node,
                    )

                    if delta < 0.0 or rng_w.random() < math.exp(-delta / max(T, 1e-12)):
                        apply_worker_move_max(
                            v=v,
                            p_from=p_from,
                            p_to=p_to,
                            part_of=local_parts,
                            out_presence=local_out_presence,
                            dest_steps=local_dest_steps,
                            active_recv_raw=local_active_recv_raw,
                            sender_hist=sender_hist,
                            loads=loads,
                            mem_step=mem_step,
                        )
                        sizes[p_from] -= 1
                        sizes[p_to] += 1
                        step_state.recompute_steps(touched_steps)
                        accepted += 1

                    T *= cooling
                    pbar.update(1)

                    if (it + 1) % post_every == 0:
                        pbar.set_postfix(
                            T=f"{T:.2e}",
                            worst=f"{worst_val:.3e}",
                            accept=f"{100.0 * accepted / (it + 1):.1f}%"
                        )

        for local_u, w in enumerate(local_parts):
            worker_of[nodes_m[local_u]] = w

        workers_per_machine.append(num_workers)

    return worker_of, workers_per_machine


# =============================================================================
# 8. Stats
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
    step_machine_load = {sk: [0.0] * num_machines for sk in step_keys}
    for v, ss in raw_recv_by_vertex.items():
        if not (0 <= v < n):
            continue
        m = machine_of[v]
        for sk, rv in ss.items():
            if sk in step_machine_load:
                step_machine_load[sk][m] += rv

    util = {}
    for sk in step_keys:
        active = sum(1 for x in step_machine_load[sk] if x > 0.0)
        util[sk] = active / num_machines if num_machines > 0 else 0.0
    return util


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


def compute_combiner_aware_final_comm_cost(
    machine_of,
    worker_of,
    out_presence,
    raw_recv_by_vertex,
    step_keys,
    n,
    c_net,
    c_mem,
):
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
    for _, hist in machine_hist.items():
        net_cost += c_net * len(hist)

    mem_cost = 0.0
    for (dst, _), hist in worker_hist_same_machine.items():
        wd = worker_of[dst]
        mem_cost += c_mem * (len(hist) - (1 if wd in hist else 0))

    return net_cost, mem_cost


def compute_combiner_aware_max_style(
    machine_of,
    worker_of,
    out_presence,
    raw_recv_by_vertex,
    step_keys,
    n,
    c_net,
    c_mem,
    c_node,
    num_machines,
    workers_per_machine,
):
    machine_hist = defaultdict(set)
    worker_hist = defaultdict(set)

    step_machine_load = {sk: [0.0] * num_machines for sk in step_keys}
    step_worker_load = {
        sk: {
            m: [0.0] * workers_per_machine[m]
            for m in range(num_machines)
            if workers_per_machine[m] > 0
        }
        for sk in step_keys
    }

    for v, ss in raw_recv_by_vertex.items():
        if not (0 <= v < n):
            continue
        m = machine_of[v]
        w = worker_of[v]
        for step_key, rv in ss.items():
            if step_key not in step_machine_load:
                continue
            step_machine_load[step_key][m] += rv
            if workers_per_machine[m] > 0:
                step_worker_load[step_key][m][w] += rv

    for u, events in out_presence.items():
        if not (0 <= u < n):
            continue
        mu = machine_of[u]
        wu = worker_of[u]

        for step_key, dst in events:
            if not (0 <= dst < n):
                continue
            md = machine_of[dst]
            wd = worker_of[dst]

            if mu != md:
                machine_hist[(dst, step_key, md)].add(mu)
            elif wu != wd:
                worker_hist[(dst, step_key, md, wd)].add(wu)

    step_machine_net = {sk: [0.0] * num_machines for sk in step_keys}
    for (dst, step_key, md), senders in machine_hist.items():
        step_machine_net[step_key][md] += len(senders)

    step_worker_mem = {
        sk: {
            m: [0.0] * workers_per_machine[m]
            for m in range(num_machines)
            if workers_per_machine[m] > 0
        }
        for sk in step_keys
    }
    for (dst, step_key, md, wd), senders in worker_hist.items():
        step_worker_mem[step_key][md][wd] += len(senders)

    total_machine_net = 0.0
    total_worker_mem = 0.0
    total_machine_bsp = 0.0
    total_worker_bsp = 0.0

    for step_key in step_keys:
        total_machine_net += c_net * sum(step_machine_net[step_key])

        machine_best = -1.0
        for m in range(num_machines):
            val = c_node * step_machine_load[step_key][m] + c_net * step_machine_net[step_key][m]
            if val > machine_best:
                machine_best = val
        total_machine_bsp += machine_best if machine_best >= 0 else 0.0

        total_worker_mem += c_mem * sum(
            sum(step_worker_mem[step_key][m]) for m in step_worker_mem[step_key]
        )

        for m in range(num_machines):
            nw = workers_per_machine[m]
            if nw <= 0:
                continue
            worker_loads = step_worker_load[step_key][m]
            worker_mems = step_worker_mem[step_key][m]

            best = -1.0
            for w in range(nw):
                x = c_node * worker_loads[w] + c_mem * worker_mems[w]
                if x > best:
                    best = x
            total_worker_bsp += best if best >= 0 else 0.0

    return total_machine_net, total_worker_mem, total_machine_bsp, total_worker_bsp


# =============================================================================
# 9. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Pregglenator combiner-aware heaviest-machine optimizer with per-trace cache"
    )
    ap.add_argument("--traces_dir", default="../comm_traces/",
                    help="Directory containing src_*/merged.csv trace files")
    ap.add_argument("--trace_paths", nargs="+", default=None,
                    help="Explicit list of merged.csv trace files to load. Overrides --traces_dir/--max_traces.")
    ap.add_argument("--num_nodes", type=int, default=None,
                    help="Graph size N. Inferred if omitted.")
    ap.add_argument("--num_machines", type=int, default=15)
    ap.add_argument("--nodes_per_machine", type=int, default=-1)
    ap.add_argument("--workers_per_machine", type=int, default=4,
                    help="Number of workers per machine. If set, overrides nodes_per_worker.")
    ap.add_argument("--nodes_per_worker", type=int, default=-1)

    ap.add_argument("--max_traces", type=int, default=6,
                    help="Maximum number of trace files to load when using --traces_dir. Loads all if <=0.")

    ap.add_argument("--c_net", type=float, default=1e-9,
                    help="Cost per combined cross-machine delivery")
    ap.add_argument("--c_mem", type=float, default=1e-9,
                    help="Cost per combined cross-worker same-machine delivery")
    ap.add_argument("--c_proc", type=float, default=None,
                    help="Alias for c_mem for backward compatibility")
    ap.add_argument("--c_node", type=float, default=1.37e-7,
                    help="Cost weight for compute bottleneck based on raw recv load")

    ap.add_argument("--sa_iters", type=int, default=500000,
                    help="Machine-level SA iterations")
    ap.add_argument("--worker_sa_iters", type=int, default=300000,
                    help="Worker-level SA iterations per machine")
    ap.add_argument("--t_start", type=float, default=None,
                    help="Machine SA start temperature. Auto-sampled if omitted.")
    ap.add_argument("--t_end", type=float, default=1e-4,
                    help="Shared SA end temperature")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--machine_init", choices=["modulo", "greedy", "metis"], default="modulo",
                    help="Machine initialization mode")
    ap.add_argument("--worker_init", choices=["greedy", "metis"], default="greedy",
                    help="Worker initialization mode inside each machine")

    ap.add_argument("--output", default="pregglenator_combiner_heaviest_machine_cached.json",
                    help="Output JSON path")

    args = ap.parse_args()

    if args.c_proc is not None:
        args.c_mem = args.c_proc

    print("Loading combiner-aware traces...")
    load_t0 = time.perf_counter()
    out_presence, raw_recv_by_vertex, pair_presence, step_keys, task_names = load_comm_traces_combiner_aware(
        args.traces_dir,
        args.max_traces,
        args.trace_paths,
    )
    print(f"  trace loading took     : {time.perf_counter() - load_t0:.2f}s")

    n = infer_num_nodes(pair_presence, raw_recv_by_vertex, out_presence, args.num_nodes)
    print(f"  tasks                 : {len(task_names)}")
    print(f"  distinct task-steps   : {len(step_keys)}")
    print(f"  nodes                 : {n:,}")

    if args.nodes_per_machine < 0:
        args.nodes_per_machine = math.ceil(n / args.num_machines)
        print(f"  nodes_per_machine     : {args.nodes_per_machine} (auto)")
    if args.workers_per_machine > 0:
        args.nodes_per_worker = max(1, math.ceil(args.nodes_per_machine / args.workers_per_machine))
        print(f"  workers_per_machine   : {args.workers_per_machine} (explicit)")
        print(f"  nodes_per_worker      : {args.nodes_per_worker} (derived)")
    elif args.nodes_per_worker < 0:
        args.nodes_per_worker = max(1, math.ceil(args.nodes_per_machine / 4))
        print(f"  nodes_per_worker      : {args.nodes_per_worker} (auto)")

    print("Building combiner-aware undirected adjacency...")
    und_adj = symmetrize_presence_to_undirected(pair_presence, n)
    vweights = build_activity_vweights(raw_recv_by_vertex, n)

    print("Initializing machine partition...")
    if args.machine_init == "modulo":
        machine_of = [u % args.num_machines for u in range(n)]
    elif args.machine_init == "greedy":
        machine_of = greedy_balanced_partition(
            und_adj,
            args.num_machines,
            args.nodes_per_machine,
            seed=args.seed,
        )
    else:
        machine_of = metis_or_fallback(
            und_adj,
            args.num_machines,
            args.nodes_per_machine,
            vweights=vweights,
            seed=args.seed,
            label="machine-init",
        )

    machine_of = repair_capacity(und_adj, machine_of, args.nodes_per_machine, args.num_machines)

    init_cut = compute_cut_weight(und_adj, machine_of)
    print(f"  initial machine cut   : {init_cut:,.1f}")

    active_recv_raw = build_active_recv_raw(raw_recv_by_vertex, n)
    dest_steps = build_dest_steps(raw_recv_by_vertex, n)

    if args.t_start is None:
        print("Sampling initial machine temperature...")
        loads_s = _initial_raw_loads(machine_of, active_recv_raw, args.num_machines)
        sender_hist_s = build_sender_part_histograms(machine_of, out_presence, n)
        net_s = build_machine_net_state(machine_of, sender_hist_s, step_keys, args.num_machines)
        step_vertices_s = build_step_vertices(out_presence, dest_steps, n)
        rng_s = random.Random(args.seed)

        t_start = sample_temperature_machine_max(
            part_of=machine_of,
            out_presence=out_presence,
            dest_steps=dest_steps,
            active_recv_raw=active_recv_raw,
            sender_hist=sender_hist_s,
            loads=loads_s,
            net_step=net_s,
            step_vertices=step_vertices_s,
            capacity=args.nodes_per_machine,
            c_net=args.c_net,
            c_node=args.c_node,
            rng=rng_s,
            k=min(2000, max(500, n // 4)),
        )
        print(f"  T_start               : {t_start:.4f}")
    else:
        t_start = args.t_start

    print(
        f"Running machine SA ({args.sa_iters:,} iterations, "
        f"T {t_start:.4f} -> {args.t_end})..."
    )
    machine_of = sa_optimize_machine_max_targeted(
        part_of=machine_of,
        out_presence=out_presence,
        dest_steps=dest_steps,
        active_recv_raw=active_recv_raw,
        num_parts=args.num_machines,
        capacity=args.nodes_per_machine,
        n=n,
        c_net=args.c_net,
        c_node=args.c_node,
        t_start=t_start,
        t_end=args.t_end,
        sa_iters=args.sa_iters,
        seed=args.seed,
        label="machine-SA",
    )

    final_cut = compute_cut_weight(und_adj, machine_of)
    print(f"  final machine cut     : {final_cut:,.1f} (delta {final_cut - init_cut:+,.1f})")

    if args.worker_sa_iters > 0:
        print(f"Assigning workers with {args.worker_init} init + SA ({args.worker_sa_iters:,} iters/machine)...")
    else:
        print(f"Assigning workers with {args.worker_init} init only...")

    worker_of, workers_per_machine = assign_workers_sa_combiner_max_targeted(
        und_adj=und_adj,
        machine_of=machine_of,
        out_presence=out_presence,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        nodes_per_worker=args.nodes_per_worker,
        c_mem=args.c_mem,
        c_node=args.c_node,
        worker_sa_iters=args.worker_sa_iters,
        t_end=args.t_end,
        seed=args.seed,
        worker_init=args.worker_init,
    )

    print("Computing final stats...")
    ss_util = compute_superstep_utilisation(
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        step_keys=step_keys,
        n=n,
    )
    avg_util = sum(ss_util.values()) / len(step_keys) if step_keys else 0.0

    final_net_cost, final_mem_cost = compute_combiner_aware_final_comm_cost(
        machine_of=machine_of,
        worker_of=worker_of,
        out_presence=out_presence,
        raw_recv_by_vertex=raw_recv_by_vertex,
        step_keys=step_keys,
        n=n,
        c_net=args.c_net,
        c_mem=args.c_mem,
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

    max_style_net, max_style_mem, max_style_machine_bsp, max_style_worker_bsp = \
        compute_combiner_aware_max_style(
            machine_of=machine_of,
            worker_of=worker_of,
            out_presence=out_presence,
            raw_recv_by_vertex=raw_recv_by_vertex,
            step_keys=step_keys,
            n=n,
            c_net=args.c_net,
            c_mem=args.c_mem,
            c_node=args.c_node,
            num_machines=args.num_machines,
            workers_per_machine=workers_per_machine,
        )

    assignment = {
        str(u): {
            "machine": int(machine_of[u]),
            "worker": int(worker_of[u]),
        }
        for u in range(n)
    }

    stats = {
        "version": "pregglenator_combiner_heaviest_machine_cached_v1",
        "trace_cache_version": TRACE_CACHE_VERSION,
        "combiner_model": {
            "machine_level": "one combined transfer per (task_step, dst_vertex, sender_machine)",
            "worker_level": "one combined same-machine cross-worker transfer per (task_step, dst_vertex, sender_worker)",
            "compute_model": "raw receive volume bottleneck per task_step",
            "optimized_target": "minimize heaviest machine runtime proxy per task_step",
        },
        "num_nodes": n,
        "num_tasks": len(task_names),
        "num_task_steps": len(step_keys),
        "num_machines": args.num_machines,
        "nodes_per_machine": args.nodes_per_machine,
        "nodes_per_worker": args.nodes_per_worker,
        "workers_per_machine": workers_per_machine,
        "machine_init": args.machine_init,
        "worker_init": args.worker_init,
        "machine_cut_weight_presence_based": final_cut,
        "avg_task_step_machine_utilisation": avg_util,
        "final_net_cost_combiner_aware": final_net_cost,
        "final_mem_cost_combiner_aware": final_mem_cost,
        "final_compute_bottleneck_raw": final_compute_cost,
        "max_style_machine_net_cost": max_style_net,
        "max_style_worker_mem_cost": max_style_mem,
        "max_style_machine_bsp": max_style_machine_bsp,
        "max_style_worker_bsp": max_style_worker_bsp,
        "total_objective_simple": final_net_cost + final_mem_cost + final_compute_cost,
        "total_objective_max_style": max_style_machine_bsp + max_style_worker_bsp,
        "worker_balance": {
            str(m): wb for m, wb in worker_balance.items()
        },
        "cost_params": {
            "c_net": args.c_net,
            "c_mem": args.c_mem,
            "c_node": args.c_node,
        },
        "trace_inputs": args.trace_paths if args.trace_paths else {
            "traces_dir": args.traces_dir,
            "max_traces": args.max_traces,
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
    print(f"  final_net_cost_combiner_aware     : {final_net_cost:,.6f}")
    print(f"  final_mem_cost_combiner_aware     : {final_mem_cost:,.6f}")
    print(f"  final_compute_bottleneck_raw      : {final_compute_cost:,.6f}")
    print(f"  total_objective_simple            : {final_net_cost + final_mem_cost + final_compute_cost:,.6f}")
    print(f"  max_style_machine_bsp             : {max_style_machine_bsp:,.6f}")
    print(f"  max_style_worker_bsp              : {max_style_worker_bsp:,.6f}")
    print(f"  total_objective_max_style         : {max_style_machine_bsp + max_style_worker_bsp:,.6f}")

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