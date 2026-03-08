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


# =============================================================================
# 3. Joint worker->machine->time objective
# =============================================================================

def _bin_to_machine_worker(bin_idx, workers_per_machine):
    return bin_idx // workers_per_machine, bin_idx % workers_per_machine


def _initial_worker_loads(bin_of, active_recv_raw, num_bins):
    loads = defaultdict(lambda: [0.0] * num_bins)
    for v, ss_list in active_recv_raw.items():
        b = bin_of[v]
        for step_key, rv in ss_list:
            loads[step_key][b] += rv
    return dict(loads)


def compute_worker_machine_time_objective(
    worker_of,
    machine_of,
    raw_recv_by_vertex,
    num_machines,
    workers_per_machine,
    step_keys,
    n,
    c_node,
):
    """
    With communication terms omitted, the paper-style nested objective

      sum_s max_m max_{w in W(m)} c_node * load(w, s)

    is equivalent to

      sum_s c_node * max_global_worker_load(s).

    We still compute it through worker loads so it matches the worker->machine->time view.
    """
    total = 0.0
    for step_key in step_keys:
        max_worker = 0.0
        worker_loads = defaultdict(float)

        for v, ss in raw_recv_by_vertex.items():
            if v < n and step_key in ss:
                key = (machine_of[v], worker_of[v])
                worker_loads[key] += ss[step_key]

        if worker_loads:
            max_worker = max(worker_loads.values())

        total += c_node * max_worker

    return total


def objective_delta_for_placement(v, bin_idx, bin_loads, active_recv_raw):
    """
    Delta in unscaled objective if vertex v were added to worker bin bin_idx.
    Objective is sum_step max_worker_load(step).
    """
    delta = 0.0
    for step_key, rv in active_recv_raw.get(v, []):
        sl = bin_loads[step_key]
        old_max = max(sl)
        new_val = sl[bin_idx] + rv
        new_max = new_val if new_val > old_max else old_max
        delta += (new_max - old_max)
    return delta


def objective_delta_for_move(v, b_from, b_to, bin_loads, active_recv_raw):
    """
    Exact unscaled delta if v moves from worker bin b_from to b_to.
    Objective is sum_step max_worker_load(step).
    """
    delta = 0.0
    for step_key, rv in active_recv_raw.get(v, []):
        sl = bin_loads[step_key]
        old_max = max(sl)

        sl[b_from] -= rv
        sl[b_to] += rv
        new_max = max(sl)
        sl[b_from] += rv
        sl[b_to] -= rv

        delta += (new_max - old_max)
    return delta


def greedy_joint_assignment(
    active_recv_raw,
    num_machines,
    workers_per_machine,
    nodes_per_machine,
    nodes_per_worker,
    n,
    seed=42,
):
    """
    Joint greedy worker assignment.

    Each worker is treated as a bin. The objective is the sum, over task-steps,
    of the maximum worker raw recv load. Since communication terms are zero,
    this is equivalent to the paper-style nested max:
        sum_s max_m max_w load(w, s)

    Hard constraints:
      - machine size <= nodes_per_machine
      - worker size <= nodes_per_worker
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

    num_bins = num_machines * workers_per_machine
    bin_of = [-1] * n
    machine_of = [-1] * n
    worker_of = [-1] * n

    machine_sizes = [0] * num_machines
    worker_sizes = [[0] * workers_per_machine for _ in range(num_machines)]
    bin_loads = defaultdict(lambda: [0.0] * num_bins)

    with tqdm(total=n, desc="Greedy joint worker assignment", unit="vertex", mininterval=0.5) as pbar:
        for v in vertices:
            best_bin = None
            best_key = None

            for m in range(num_machines):
                if machine_sizes[m] >= nodes_per_machine:
                    continue
                for w in range(workers_per_machine):
                    if worker_sizes[m][w] >= nodes_per_worker:
                        continue

                    b = m * workers_per_machine + w
                    d = objective_delta_for_placement(v, b, bin_loads, active_recv_raw)
                    key = (d, machine_sizes[m], worker_sizes[m][w], m, w)

                    if best_key is None or key < best_key:
                        best_key = key
                        best_bin = b

            if best_bin is None:
                raise RuntimeError("Capacity infeasible during greedy joint assignment.")

            m, w = _bin_to_machine_worker(best_bin, workers_per_machine)
            bin_of[v] = best_bin
            machine_of[v] = m
            worker_of[v] = w
            machine_sizes[m] += 1
            worker_sizes[m][w] += 1

            for step_key, rv in active_recv_raw.get(v, []):
                bin_loads[step_key][best_bin] += rv

            pbar.update(1)

    return machine_of, worker_of, bin_of


def greedy_joint_local_refine(
    machine_of,
    worker_of,
    bin_of,
    active_recv_raw,
    num_machines,
    workers_per_machine,
    nodes_per_machine,
    nodes_per_worker,
    n,
    max_passes=3,
):
    """
    Deterministic improving local search over worker bins.
    Only accepts improving moves.
    """
    machine_of = list(machine_of)
    worker_of = list(worker_of)
    bin_of = list(bin_of)

    num_bins = num_machines * workers_per_machine

    machine_sizes = [0] * num_machines
    worker_sizes = [[0] * workers_per_machine for _ in range(num_machines)]
    for v in range(n):
        m = machine_of[v]
        w = worker_of[v]
        machine_sizes[m] += 1
        worker_sizes[m][w] += 1

    bin_loads = _initial_worker_loads(bin_of, active_recv_raw, num_bins)

    for pass_idx in range(max_passes):
        improved = 0
        with tqdm(total=n, desc=f"Refine pass {pass_idx + 1}", unit="vertex", mininterval=0.5) as pbar:
            for v in range(n):
                b_from = bin_of[v]
                m_from = machine_of[v]
                w_from = worker_of[v]

                best_delta = 0.0
                best_to = None

                for m_to in range(num_machines):
                    machine_after = machine_sizes[m_to] + (0 if m_to == m_from else 1)
                    if machine_after > nodes_per_machine:
                        continue

                    for w_to in range(workers_per_machine):
                        if m_to == m_from and w_to == w_from:
                            continue

                        worker_after = worker_sizes[m_to][w_to] + (0 if (m_to == m_from and w_to == w_from) else 1)
                        if worker_after > nodes_per_worker:
                            continue

                        b_to = m_to * workers_per_machine + w_to
                        delta = objective_delta_for_move(v, b_from, b_to, bin_loads, active_recv_raw)

                        if delta < best_delta:
                            best_delta = delta
                            best_to = b_to

                if best_to is not None:
                    m_to, w_to = _bin_to_machine_worker(best_to, workers_per_machine)

                    for step_key, rv in active_recv_raw.get(v, []):
                        bin_loads[step_key][b_from] -= rv
                        bin_loads[step_key][best_to] += rv

                    machine_sizes[m_from] -= 1
                    worker_sizes[m_from][w_from] -= 1
                    machine_sizes[m_to] += 1
                    worker_sizes[m_to][w_to] += 1

                    bin_of[v] = best_to
                    machine_of[v] = m_to
                    worker_of[v] = w_to
                    improved += 1

                pbar.update(1)

        print(f"  refine pass {pass_idx + 1}: improved moves = {improved}")
        if improved == 0:
            break

    return machine_of, worker_of, bin_of


# =============================================================================
# 4. Stats
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
    workers_per_machine_list,
    num_machines,
    step_keys,
    n,
):
    stats = {}

    for m in range(num_machines):
        nw = workers_per_machine_list[m]
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


def compute_machine_balance(
    machine_of,
    raw_recv_by_vertex,
    num_machines,
    step_keys,
    n,
):
    sum_max = 0.0
    sum_avg = 0.0

    for step_key in step_keys:
        ml = [0.0] * num_machines
        for v, ss in raw_recv_by_vertex.items():
            if v < n and step_key in ss:
                ml[machine_of[v]] += ss[step_key]

        sum_max += max(ml) if ml else 0.0
        sum_avg += (sum(ml) / num_machines) if num_machines > 0 else 0.0

    ratio = (sum_max / sum_avg) if sum_avg > 0 else 1.0
    return {
        "sum_max_machine_load": sum_max,
        "sum_avg_machine_load": sum_avg,
        "imbalance_ratio": ratio,
    }


# =============================================================================
# 5. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Pregglenator compute-only joint optimizer: raw recv load worker->machine->time objective"
    )
    ap.add_argument("--traces_dir", default="../comm_traces/",
                    help="Directory containing src_*/merged.csv trace files")
    ap.add_argument("--trace_paths", nargs="+", default=None,
                    help="Explicit list of merged.csv trace files to load. Overrides --traces_dir/--max_traces.")
    ap.add_argument("--num_nodes", type=int, default=None,
                    help="Graph size N. Inferred if omitted.")
    ap.add_argument("--num_machines", type=int, default=4)
    ap.add_argument("--nodes_per_machine", type=int, default=-1)
    ap.add_argument("--workers_per_machine", type=int, default=15,
                    help="Number of workers per machine. If set, overrides nodes_per_worker.")
    ap.add_argument("--nodes_per_worker", type=int, default=-1)

    ap.add_argument("--max_traces", type=int, default=6,
                    help="Maximum number of trace files to load when using --traces_dir. Loads all if <=0.")

    ap.add_argument("--c_node", type=float, default=1.37e-07,
                    help="Cost weight for compute bottleneck based on raw recv load")

    ap.add_argument("--joint_refine_passes", type=int, default=2,
                    help="Number of greedy local-refinement passes after joint worker assignment")

    ap.add_argument("--output", default="pregglenator_joint_4m.json",
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
        workers_per_machine = args.workers_per_machine
        args.nodes_per_worker = max(1, math.ceil(args.nodes_per_machine / workers_per_machine))
        print(f"  workers_per_machine   : {workers_per_machine} (explicit)")
        print(f"  nodes_per_worker      : {args.nodes_per_worker} (derived from workers_per_machine)")
    elif args.nodes_per_worker > 0:
        workers_per_machine = max(1, math.ceil(args.nodes_per_machine / args.nodes_per_worker))
        print(f"  workers_per_machine   : {workers_per_machine} (derived from nodes_per_worker)")
        print(f"  nodes_per_worker      : {args.nodes_per_worker} (explicit)")
    else:
        workers_per_machine = 4
        args.nodes_per_worker = max(1, math.ceil(args.nodes_per_machine / workers_per_machine))
        print(f"  workers_per_machine   : {workers_per_machine} (default)")
        print(f"  nodes_per_worker      : {args.nodes_per_worker} (derived)")

    total_worker_capacity = args.num_machines * workers_per_machine * args.nodes_per_worker
    total_machine_capacity = args.num_machines * args.nodes_per_machine
    if total_machine_capacity < n:
        raise RuntimeError(
            f"Insufficient machine capacity: {total_machine_capacity} < {n}"
        )
    if total_worker_capacity < n:
        raise RuntimeError(
            f"Insufficient worker capacity: {total_worker_capacity} < {n}"
        )

    active_recv_raw = build_active_recv_raw(raw_recv_by_vertex, n)

    print("Greedy joint worker assignment...")
    machine_of, worker_of, bin_of = greedy_joint_assignment(
        active_recv_raw=active_recv_raw,
        num_machines=args.num_machines,
        workers_per_machine=workers_per_machine,
        nodes_per_machine=args.nodes_per_machine,
        nodes_per_worker=args.nodes_per_worker,
        n=n,
        seed=42,
    )

    init_objective = compute_worker_machine_time_objective(
        worker_of=worker_of,
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        workers_per_machine=workers_per_machine,
        step_keys=step_keys,
        n=n,
        c_node=args.c_node,
    )
    print(f"  initial joint objective          : {init_objective:,.6f}")

    if args.joint_refine_passes > 0:
        print(f"Refining joint assignment ({args.joint_refine_passes} pass(es))...")
        machine_of, worker_of, bin_of = greedy_joint_local_refine(
            machine_of=machine_of,
            worker_of=worker_of,
            bin_of=bin_of,
            active_recv_raw=active_recv_raw,
            num_machines=args.num_machines,
            workers_per_machine=workers_per_machine,
            nodes_per_machine=args.nodes_per_machine,
            nodes_per_worker=args.nodes_per_worker,
            n=n,
            max_passes=args.joint_refine_passes,
        )

    final_objective = compute_worker_machine_time_objective(
        worker_of=worker_of,
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        num_machines=args.num_machines,
        workers_per_machine=workers_per_machine,
        step_keys=step_keys,
        n=n,
        c_node=args.c_node,
    )
    print(f"  final joint objective            : {final_objective:,.6f} (delta {final_objective - init_objective:+,.6f})")

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

    workers_per_machine_list = [workers_per_machine] * args.num_machines

    worker_balance = compute_worker_balance(
        worker_of=worker_of,
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
        workers_per_machine_list=workers_per_machine_list,
        num_machines=args.num_machines,
        step_keys=step_keys,
        n=n,
    )

    machine_balance = compute_machine_balance(
        machine_of=machine_of,
        raw_recv_by_vertex=raw_recv_by_vertex,
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
        "version": "pregglenator_compute_only_joint_rawrecv_v1",
        "trace_cache_version": TRACE_CACHE_VERSION,
        "objective_model": {
            "worker_level": "raw recv load per worker per task_step",
            "machine_level": "max worker raw recv load within each machine per task_step",
            "query_level": "sum over task_steps of max machine cost",
            "equivalent_compute_only_form": "sum over task_steps of max worker raw recv load",
            "communication_terms": "omitted intentionally because fitted c_net = c_proc = 0",
        },
        "optimizer": {
            "assignment": "joint greedy worker-bin placement + local refinement",
            "bin_definition": "each bin is a (machine, worker) pair",
        },
        "num_nodes": n,
        "num_tasks": len(task_names),
        "num_task_steps": len(step_keys),
        "num_machines": args.num_machines,
        "nodes_per_machine": args.nodes_per_machine,
        "workers_per_machine": workers_per_machine_list,
        "nodes_per_worker": args.nodes_per_worker,
        "avg_task_step_machine_utilisation": avg_util,
        "final_compute_bottleneck_raw": final_objective,
        "total_objective": final_objective,
        "machine_balance": machine_balance,
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
    print(f"  final_compute_bottleneck_raw      : {final_objective:,.6f}")
    print(f"  total_objective                   : {final_objective:,.6f}")
    print(
        f"  machine balance ratio             : "
        f"{machine_balance['imbalance_ratio']:.3f} "
        f"(sum_max={machine_balance['sum_max_machine_load']:.1f}, "
        f"sum_avg={machine_balance['sum_avg_machine_load']:.1f})"
    )

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