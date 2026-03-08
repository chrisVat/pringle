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
from collections import defaultdict

from tqdm import tqdm

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False


TRACE_CACHE_VERSION = "compute_only_trace_cache_v2_greedy"


# =============================================================================
# 1. Loading traces
# =============================================================================

def _load_single_trace_compute_only(path):
    """
    Load one src_*/merged.csv and return per-trace data needed for compute-only
    optimization.

    Returns
    -------
    task_name : str
    raw_recv_local : dict[int, dict[step_key, float]]
        raw_recv_local[v][(task_name, superstep)] = raw recv load at vertex v
    step_keys_local : list[tuple[str, int]]
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"Trace file not found: {path}")

    task_name = os.path.basename(os.path.dirname(path))
    cache_path = path + ".compute_only.pkl"
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
                return result
            else:
                print(f"    cache stale for {task_name} — rebuilding...")
        except Exception as e:
            print(f"    cache load failed for {task_name} ({e}) — rebuilding...")

    raw_recv_local = defaultdict(lambda: defaultdict(float))
    step_keys_set = set()

    if _HAVE_PANDAS:
        df = pd.read_csv(
            path,
            dtype={
                "superstep": "int32",
                "src_vertex": "int32",
                "dst_vertex": "int32",
                "count": "float32",
            },
        )
        df = df[(df["src_vertex"] != df["dst_vertex"]) & (df["count"] > 0)]

        recv_grouped = df.groupby(
            ["superstep", "dst_vertex"],
            sort=False
        )["count"].sum()

        unique_steps = sorted(int(s) for s in df["superstep"].unique())
        step_key_of = {s: (task_name, s) for s in unique_steps}
        step_keys_set = set(step_key_of.values())

        print(f"    pandas: {len(df):,} rows -> {len(recv_grouped):,} unique (step,dst) recv pairs")

        for (s, v), csum in tqdm(
            recv_grouped.items(),
            total=len(recv_grouped),
            desc=f"    recv index {task_name}",
            unit="pair",
            mininterval=0.5,
            leave=False,
        ):
            raw_recv_local[int(v)][step_key_of[int(s)]] += float(csum)

    else:
        local_recv = defaultdict(float)
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            ci = {name: i for i, name in enumerate(header)}
            si, vi, ci_ = ci["superstep"], ci["dst_vertex"], ci["count"]
            ui = ci["src_vertex"]

            for row in tqdm(
                reader,
                desc=f"    reading {task_name}",
                unit="row",
                mininterval=0.5,
                leave=False,
            ):
                s = int(row[si])
                u = int(row[ui])
                v = int(row[vi])
                c = float(row[ci_])
                if u != v and c > 0:
                    local_recv[(s, v)] += c

        unique_steps = sorted({int(s) for (s, _) in local_recv.keys()})
        step_key_of = {s: (task_name, s) for s in unique_steps}
        step_keys_set = set(step_key_of.values())

        for (s, v), csum in tqdm(
            local_recv.items(),
            total=len(local_recv),
            desc=f"    recv index {task_name}",
            unit="pair",
            mininterval=0.5,
            leave=False,
        ):
            raw_recv_local[int(v)][step_key_of[int(s)]] += float(csum)

    step_keys_local = sorted(step_keys_set, key=lambda x: (x[0], x[1]))
    result = (
        task_name,
        {v: dict(ss) for v, ss in raw_recv_local.items()},
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
        print(f"    cache write failed for {task_name} ({e}) — continuing without cache")

    return result


def load_comm_traces_compute_only(traces_dir, max_traces=-1, trace_paths=None):
    """
    Returns
    -------
    raw_recv_by_vertex : dict[int, dict[step_key, float]]
    step_keys : list[tuple[str, int]]
    task_names : list[str]
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

    raw_recv_by_vertex = defaultdict(lambda: defaultdict(float))
    step_keys_set = set()
    task_names = []

    for path in tqdm(paths, desc="Reading traces", unit="file"):
        task_name, raw_recv_local, step_keys_local = _load_single_trace_compute_only(path)
        task_names.append(task_name)

        for v, ss in raw_recv_local.items():
            dst_map = raw_recv_by_vertex[v]
            for step_key, csum in ss.items():
                dst_map[step_key] += csum

        for step_key in step_keys_local:
            step_keys_set.add(step_key)

    step_keys = sorted(step_keys_set, key=lambda x: (x[0], x[1]))
    return {v: dict(ss) for v, ss in raw_recv_by_vertex.items()}, step_keys, task_names


def infer_num_nodes(raw_recv_by_vertex, explicit_n=None):
    if explicit_n is not None:
        return explicit_n
    mx = max(raw_recv_by_vertex.keys(), default=-1)
    return mx + 1


# =============================================================================
# 2. Sparse per-vertex state
# =============================================================================

def build_active_recv_raw(raw_recv_by_vertex, n):
    active = {}
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n:
            active[v] = list(ss.items())
    return active


def build_total_recv_weights(raw_recv_by_vertex, n):
    w = [0.0] * n
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n:
            w[v] = sum(ss.values())
    return w


def build_peak_recv_weights(raw_recv_by_vertex, n):
    w = [0.0] * n
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n and ss:
            w[v] = max(ss.values())
    return w


def build_activity_counts(raw_recv_by_vertex, n):
    w = [0] * n
    for v, ss in raw_recv_by_vertex.items():
        if 0 <= v < n:
            w[v] = len(ss)
    return w


def _initial_raw_loads(part_of, active_recv_raw, num_parts):
    loads = defaultdict(lambda: [0.0] * num_parts)
    for v, ss_list in active_recv_raw.items():
        p = part_of[v]
        for step_key, rv in ss_list:
            loads[step_key][p] += rv
    return dict(loads)


def compute_compute_bottleneck(machine_of, raw_recv_by_vertex, num_machines, step_keys, n, c_node):
    total = 0.0
    for step_key in step_keys:
        ml = [0.0] * num_machines
        for v, ss in raw_recv_by_vertex.items():
            if v < n and step_key in ss:
                ml[machine_of[v]] += ss[step_key]
        total += c_node * max(ml)
    return total


# =============================================================================
# 3. Greedy vector balancing
# =============================================================================

def objective_delta_for_placement(v, p, part_loads, active_recv_raw):
    """
    Delta in unscaled objective if vertex v were added to partition p.
    Objective is sum_step max_machine_load(step).
    """
    delta = 0.0
    for step_key, rv in active_recv_raw.get(v, []):
        sl = part_loads[step_key]
        old_max = max(sl)
        new_val = sl[p] + rv
        new_max = new_val if new_val > old_max else old_max
        delta += (new_max - old_max)
    return delta


def objective_delta_for_move(v, p_from, p_to, part_loads, active_recv_raw):
    """
    Exact unscaled delta if v moves from p_from to p_to.
    """
    delta = 0.0
    for step_key, rv in active_recv_raw.get(v, []):
        sl = part_loads[step_key]
        old_max = max(sl)

        sl[p_from] -= rv
        sl[p_to] += rv
        new_max = max(sl)
        sl[p_from] += rv
        sl[p_to] -= rv

        delta += (new_max - old_max)
    return delta


def greedy_balanced_assignment(active_recv_raw, num_parts, capacity, n, seed=42):
    """
    Greedy vector balancing:
    - sort vertices by temporal importance
    - place each on the machine with smallest objective increase
    - respect hard capacity
    """
    rng = random.Random(seed)

    total_w = [0.0] * n
    peak_w = [0.0] * n
    act_w = [0] * n
    for v in range(n):
        ss = active_recv_raw.get(v, [])
        if ss:
            vals = [rv for _, rv in ss]
            total_w[v] = sum(vals)
            peak_w[v] = max(vals)
            act_w[v] = len(vals)

    vertices = list(range(n))
    rng.shuffle(vertices)
    vertices.sort(
        key=lambda v: (peak_w[v], total_w[v], act_w[v]),
        reverse=True,
    )

    part_of = [-1] * n
    sizes = [0] * num_parts
    part_loads = defaultdict(lambda: [0.0] * num_parts)

    with tqdm(total=n, desc="Greedy machine assignment", unit="vertex", mininterval=0.5) as pbar:
        for v in vertices:
            best_p = None
            best_key = None

            for p in range(num_parts):
                if sizes[p] >= capacity:
                    continue

                d = objective_delta_for_placement(v, p, part_loads, active_recv_raw)
                key = (d, sizes[p], p)
                if best_key is None or key < best_key:
                    best_key = key
                    best_p = p

            if best_p is None:
                raise RuntimeError("Capacity infeasible during greedy assignment.")

            part_of[v] = best_p
            sizes[best_p] += 1
            for step_key, rv in active_recv_raw.get(v, []):
                part_loads[step_key][best_p] += rv

            pbar.update(1)

    return part_of


def greedy_local_refine(
    part_of,
    active_recv_raw,
    num_parts,
    capacity,
    n,
    max_passes=3,
):
    """
    Deterministic improving local search.
    Only accepts improving moves.
    """
    part_of = list(part_of)
    sizes = [0] * num_parts
    for p in part_of:
        sizes[p] += 1

    part_loads = _initial_raw_loads(part_of, active_recv_raw, num_parts)

    for pass_idx in range(max_passes):
        improved = 0
        with tqdm(total=n, desc=f"Refine pass {pass_idx + 1}", unit="vertex", mininterval=0.5) as pbar:
            for v in range(n):
                p_from = part_of[v]
                best_delta = 0.0
                best_to = None

                for p_to in range(num_parts):
                    if p_to == p_from:
                        continue
                    if sizes[p_to] >= capacity:
                        continue

                    delta = objective_delta_for_move(v, p_from, p_to, part_loads, active_recv_raw)
                    if delta < best_delta:
                        best_delta = delta
                        best_to = p_to

                if best_to is not None:
                    for step_key, rv in active_recv_raw.get(v, []):
                        part_loads[step_key][p_from] -= rv
                        part_loads[step_key][best_to] += rv

                    part_of[v] = best_to
                    sizes[p_from] -= 1
                    sizes[best_to] += 1
                    improved += 1

                pbar.update(1)

        print(f"  refine pass {pass_idx + 1}: improved moves = {improved}")
        if improved == 0:
            break

    return part_of


# =============================================================================
# 4. Worker assignment
# =============================================================================

def build_local_raw_recv(nodes_m, old_to_local, raw_recv_by_vertex):
    local_recv = {}
    for old_v in nodes_m:
        if old_v in raw_recv_by_vertex:
            local_recv[old_to_local[old_v]] = list(raw_recv_by_vertex[old_v].items())
    return local_recv


def assign_workers_greedy_compute_only(
    machine_of,
    raw_recv_by_vertex,
    num_machines,
    nodes_per_worker,
    c_node,
    refine_passes,
):
    n = len(machine_of)
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
        local_active_recv_raw = build_local_raw_recv(nodes_m, old_to_local, raw_recv_by_vertex)

        print(f"  machine {m}: greedy worker assignment with {num_workers} workers, {len(nodes_m)} nodes")
        local_parts = greedy_balanced_assignment(
            active_recv_raw=local_active_recv_raw,
            num_parts=num_workers,
            capacity=nodes_per_worker,
            n=len(nodes_m),
            seed=1234 + m,
        )

        if refine_passes > 0:
            local_parts = greedy_local_refine(
                part_of=local_parts,
                active_recv_raw=local_active_recv_raw,
                num_parts=num_workers,
                capacity=nodes_per_worker,
                n=len(nodes_m),
                max_passes=refine_passes,
            )

        for local_u, w in enumerate(local_parts):
            worker_of[nodes_m[local_u]] = w

        workers_per_machine.append(num_workers)

    return worker_of, workers_per_machine


# =============================================================================
# 5. Stats
# =============================================================================

def compute_superstep_utilisation(machine_of, raw_recv_by_vertex, num_machines, step_keys, n):
    util = {}
    for step_key in step_keys:
        active = set()
        for v, ss in raw_recv_by_vertex.items():
            if v < n and step_key in ss:
                active.add(machine_of[v])
        util[step_key] = len(active)
    return util


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
# 6. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Pregglenator compute-only greedy optimizer: balance per-step raw recv load across machines"
    )
    ap.add_argument("--traces_dir", default="../comm_traces/",
                    help="Directory containing src_*/merged.csv trace files")
    ap.add_argument("--trace_paths", nargs="+", default=None,
                    help="Explicit list of merged.csv trace files to load. Overrides --traces_dir/--max_traces.")
    ap.add_argument("--num_nodes", type=int, default=None,
                    help="Graph size N. Inferred if omitted.")
    ap.add_argument("--num_machines", type=int, default=4)  # HERE!!!
    ap.add_argument("--nodes_per_machine", type=int, default=-1)
    ap.add_argument("--workers_per_machine", type=int, default=4,
                    help="Number of workers per machine. If set, overrides nodes_per_worker.")
    ap.add_argument("--nodes_per_worker", type=int, default=-1)

    ap.add_argument("--max_traces", type=int, default=-1,
                    help="Maximum number of trace files to load when using --traces_dir. Loads all if <=0.")

    ap.add_argument("--c_node", type=float, default=1.37e-07,
                    help="Cost weight for compute bottleneck based on raw recv load")

    ap.add_argument("--machine_refine_passes", type=int, default=4,
                    help="Number of greedy local-refinement passes after machine assignment")
    ap.add_argument("--worker_refine_passes", type=int, default=2,
                    help="Number of greedy local-refinement passes after worker assignment")

    ap.add_argument("--output", default="pregglenator_greedmax_bigguy_fulltrain_4m.json",
                    help="Output JSON path")

    args = ap.parse_args()

    print("Loading compute-only traces...")
    load_t0 = time.perf_counter()
    raw_recv_by_vertex, step_keys, task_names = load_comm_traces_compute_only(
        args.traces_dir,
        args.max_traces,
        args.trace_paths,
    )
    print(f"  trace loading took     : {time.perf_counter() - load_t0:.2f}s")

    n = infer_num_nodes(raw_recv_by_vertex, args.num_nodes)
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

    active_recv_raw = build_active_recv_raw(raw_recv_by_vertex, n)

    print("Greedy machine assignment...")
    machine_of = greedy_balanced_assignment(
        active_recv_raw=active_recv_raw,
        num_parts=args.num_machines,
        capacity=args.nodes_per_machine,
        n=n,
        seed=42,
    )

    init_compute_cost = compute_compute_bottleneck(
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        step_keys=step_keys,
        n=n,
        c_node=args.c_node,
    )
    print(f"  initial greedy objective         : {init_compute_cost:,.6f}")

    if args.machine_refine_passes > 0:
        print(f"Refining machine assignment ({args.machine_refine_passes} pass(es))...")
        machine_of = greedy_local_refine(
            part_of=machine_of,
            active_recv_raw=active_recv_raw,
            num_parts=args.num_machines,
            capacity=args.nodes_per_machine,
            n=n,
            max_passes=args.machine_refine_passes,
        )

    final_compute_cost = compute_compute_bottleneck(
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        step_keys=step_keys,
        n=n,
        c_node=args.c_node,
    )
    print(f"  final machine objective          : {final_compute_cost:,.6f} (delta {final_compute_cost - init_compute_cost:+,.6f})")

    print("Assigning workers with greedy compute-only balancing...")
    worker_of, workers_per_machine = assign_workers_greedy_compute_only(
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        nodes_per_worker=args.nodes_per_worker,
        c_node=args.c_node,
        refine_passes=args.worker_refine_passes,
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

    total_objective = compute_compute_bottleneck(
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
        "version": "pregglenator_compute_only_greedy_v1",
        "trace_cache_version": TRACE_CACHE_VERSION,
        "objective_model": {
            "machine_level": "sum over task_steps of max machine raw recv load",
            "worker_level": "sum over task_steps of max worker raw recv load within each machine",
            "communication_terms": "omitted intentionally because fitted c_net = c_proc = 0"
        },
        "optimizer": {
            "machine_assignment": "greedy sparse vector balancing + local refinement",
            "worker_assignment": "greedy sparse vector balancing + local refinement"
        },
        "num_nodes": n,
        "num_tasks": len(task_names),
        "num_task_steps": len(step_keys),
        "num_machines": args.num_machines,
        "nodes_per_machine": args.nodes_per_machine,
        "nodes_per_worker": args.nodes_per_worker,
        "workers_per_machine": workers_per_machine,
        "avg_task_step_machine_utilisation": avg_util,
        "final_compute_bottleneck_raw": total_objective,
        "total_objective": total_objective,
        "worker_balance": {
            str(m): wb for m, wb in worker_balance.items()
        },
        "cost_params": {
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
    print(f"  avg_task_step_machine_utilisation : {avg_util:.3f}")
    print(f"  final_compute_bottleneck_raw      : {total_objective:,.6f}")
    print(f"  total_objective                   : {total_objective:,.6f}")

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