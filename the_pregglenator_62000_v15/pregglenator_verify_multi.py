#!/usr/bin/env python3
"""
pregglenator_verify_cv.py

Leave-one-run-out cross-validation for the Pregglenator runtime model.

Each run is one timed execution with:
- a merged.csv trace
- a timing JSON
- a partition JSON

For each run, this script:
1. Loads the trace
2. Loads the partition
3. Computes combiner-aware raw per-step features:
      net_raw(step), proc_raw(step), compute_raw(step)
4. Aligns them to observed per-step runtime

Then it performs leave-one-run-out CV:
- fit on N-1 runs
- test on the held-out run

Model
-----
time(step) = c_bias + c_net * net + c_proc * proc + c_node * compute

Outputs
-------
- per-run proxy totals under user-supplied weights
- per-run held-out CV prediction using NNLS fit from all other runs
- aggregate CV metrics across runs

Notes
-----
- This is run-level CV, not per-step CV.
- The compute_raw feature is the max machine raw recv load in a step.
- net_raw / proc_raw are combiner-aware distinct sender-machine / sender-worker events.
"""

import argparse
import csv
import json
import math
import os
import pickle
import time
from collections import defaultdict

from tqdm import tqdm

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False

try:
    import numpy as np
    from scipy.optimize import nnls as _scipy_nnls
    _HAVE_SCIPY = True
except ImportError:
    try:
        import numpy as np
        _HAVE_SCIPY = False
    except ImportError:
        np = None
        _HAVE_SCIPY = False


TRACE_CACHE_VERSION = "verify_trace_v1"
COMP_CACHE_VERSION = "verify_comp_v1"


# =============================================================================
# 1. Loading
# =============================================================================

def load_trace(trace_path):
    """Load a single merged.csv trace file, using a pickle cache when available."""
    if not os.path.isfile(trace_path):
        raise RuntimeError(f"Trace file not found: {trace_path}")

    task_name = os.path.basename(os.path.dirname(trace_path))
    cache_path = trace_path + ".verify.pkl"
    trace_mtime = os.path.getmtime(trace_path)

    if os.path.isfile(cache_path):
        try:
            t_load = time.perf_counter()
            with open(cache_path, "rb") as fh:
                cached_key, result = pickle.load(fh)
            if cached_key == (TRACE_CACHE_VERSION, trace_mtime):
                out_presence, raw_recv_by_vertex, step_keys = result
                print(
                    f"    trace cache hit ({time.perf_counter() - t_load:.1f}s)  |  "
                    f"{task_name}  |  {len(step_keys)} steps"
                )
                return out_presence, raw_recv_by_vertex, step_keys
            print(f"    trace cache stale for {task_name} — rebuilding...")
        except Exception as e:
            print(f"    trace cache load failed for {task_name} ({e}) — rebuilding...")

    out_presence = defaultdict(list)
    raw_recv_by_vertex = defaultdict(lambda: defaultdict(float))
    step_keys_set = set()

    if _HAVE_PANDAS:
        df = pd.read_csv(
            trace_path,
            dtype={
                "superstep": "int32",
                "src_vertex": "int32",
                "dst_vertex": "int32",
                "count": "float32",
            },
        )
        df = df[(df["src_vertex"] != df["dst_vertex"]) & (df["count"] > 0)]

        grouped = df.groupby(
            ["superstep", "src_vertex", "dst_vertex"],
            sort=False
        )["count"].sum()
        print(f"    pandas: {len(df):,} rows -> {len(grouped):,} unique (s,u,v) pairs")

        recv_grouped = df.groupby(
            ["superstep", "dst_vertex"],
            sort=False
        )["count"].sum()

        for (s, v), csum in tqdm(
            recv_grouped.items(),
            total=len(recv_grouped),
            desc="    Building recv index",
            unit="pair",
            mininterval=0.5,
            leave=False,
        ):
            raw_recv_by_vertex[int(v)][(task_name, int(s))] += float(csum)

        presence_df = grouped.reset_index()[["superstep", "src_vertex", "dst_vertex"]]
        step_key_of = {
            int(s): (task_name, int(s))
            for s in presence_df["superstep"].unique()
        }
        step_keys_set = set(step_key_of.values())

        for s, u, v in tqdm(
            presence_df.itertuples(index=False, name=None),
            total=len(presence_df),
            desc="    Building presence index",
            unit="edge",
            mininterval=0.5,
            leave=False,
        ):
            out_presence[int(u)].append((step_key_of[int(s)], int(v)))
    else:
        local_counts = defaultdict(float)

        with open(trace_path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            cols = {name: i for i, name in enumerate(header)}
            si = cols["superstep"]
            ui = cols["src_vertex"]
            vi = cols["dst_vertex"]
            ci = cols["count"]

            for row in tqdm(reader, desc="    Reading trace", unit="row", leave=False):
                s = int(row[si])
                u = int(row[ui])
                v = int(row[vi])
                c = float(row[ci])
                if u != v and c > 0:
                    local_counts[(s, u, v)] += c

        unique_steps = sorted({int(s) for (s, _, _) in local_counts.keys()})
        step_key_of = {s: (task_name, s) for s in unique_steps}
        step_keys_set = set(step_key_of.values())

        for (s, u, v), csum in tqdm(
            local_counts.items(),
            total=len(local_counts),
            desc="    Building index",
            unit="pair",
            mininterval=0.5,
            leave=False,
        ):
            step_key = step_key_of[int(s)]
            out_presence[int(u)].append((step_key, int(v)))
            raw_recv_by_vertex[int(v)][step_key] += float(csum)

    step_keys = sorted(step_keys_set, key=lambda x: (x[0], x[1]))
    out_presence = dict(out_presence)
    raw_recv_by_vertex = {v: dict(ss) for v, ss in raw_recv_by_vertex.items()}

    try:
        t_write = time.perf_counter()
        with open(cache_path, "wb") as fh:
            pickle.dump(
                ((TRACE_CACHE_VERSION, trace_mtime), (out_presence, raw_recv_by_vertex, step_keys)),
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"    trace cache written ({time.perf_counter() - t_write:.1f}s)")
    except Exception as e:
        print(f"    trace cache write failed ({e})")

    return out_presence, raw_recv_by_vertex, step_keys


def load_partition(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"Partition file not found: {path}")

    with open(path) as fh:
        data = json.load(fh)

    assignment = data["assignment"]
    machine_of = {}
    worker_of = {}

    for node_str, info in assignment.items():
        node = int(node_str)
        machine_of[node] = info["machine"]
        worker_of[node] = info["worker"]

    return machine_of, worker_of


def load_timings(path):
    """
    Load a timing JSON file.

    Returns
    -------
    times : {superstep: elapsed_seconds}
    actual_msgs : {superstep: {"msgs", "cross_worker", "cross_machine"}}
    total_time : float | None
    source : Any
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"Timing file not found: {path}")

    with open(path) as fh:
        data = json.load(fh)

    times = {}
    actual_msgs = {}
    for entry in data.get("supersteps", []):
        s = entry["superstep"]
        times[s] = entry["time"]
        actual_msgs[s] = {
            "msgs": entry.get("msgs", 0),
            "cross_worker": entry.get("cross_worker", 0),
            "cross_machine": entry.get("cross_machine", 0),
        }

    return times, actual_msgs, data.get("total_time"), data.get("source")


# =============================================================================
# 2. Feature computation
# =============================================================================

def compute_raw_components(out_presence, raw_recv_by_vertex, machine_of, worker_of, step_keys):
    """
    Precompute the three unscaled cost components per step.

    Returns
    -------
    net_raw : {step_key: float}
    proc_raw : {step_key: float}
    compute_raw : {step_key: float}
    machine_loads : {step_key: list[float]}
    """
    num_machines = (max(machine_of.values()) + 1) if machine_of else 1

    machine_hist = defaultdict(set)   # (dst, step_key) -> {sender_machine, ...}
    worker_hist = defaultdict(set)    # (dst, step_key) -> {sender_worker, ...}

    total_events = sum(len(v) for v in out_presence.values())

    with tqdm(
        total=len(out_presence),
        desc="    Sender histograms",
        unit="vertex",
        mininterval=0.5,
        leave=False,
    ) as pbar:
        for u, events in out_presence.items():
            mu = machine_of.get(u)
            wu = worker_of.get(u)

            if mu is not None:
                for step_key, dst in events:
                    md = machine_of.get(dst)
                    if md is None:
                        continue
                    if mu != md:
                        machine_hist[(dst, step_key)].add(mu)
                    else:
                        worker_hist[(dst, step_key)].add(wu)

            pbar.update(1)

    print(
        f"    histograms: {len(machine_hist):,} cross-machine pairs  |  "
        f"{len(worker_hist):,} cross-worker pairs  |  {total_events:,} presence events"
    )

    net_raw = defaultdict(float)
    proc_raw = defaultdict(float)

    for (dst, step_key), hist in tqdm(
        machine_hist.items(),
        desc="    Net cost aggregation",
        unit="pair",
        mininterval=0.5,
        leave=False,
    ):
        net_raw[step_key] += len(hist)

    for (dst, step_key), hist in tqdm(
        worker_hist.items(),
        desc="    Proc cost aggregation",
        unit="pair",
        mininterval=0.5,
        leave=False,
    ):
        wd = worker_of.get(dst)
        proc_raw[step_key] += len(hist) - (1 if wd in hist else 0)

    step_machine_loads = defaultdict(lambda: [0.0] * num_machines)
    for v, ss in tqdm(
        raw_recv_by_vertex.items(),
        desc="    Compute bottleneck",
        unit="vertex",
        mininterval=0.5,
        leave=False,
    ):
        m = machine_of.get(v)
        if m is None:
            continue
        for step_key, load in ss.items():
            step_machine_loads[step_key][m] += load

    compute_raw = {}
    machine_loads = {}
    for step_key in step_keys:
        ml = step_machine_loads.get(step_key, [0.0] * num_machines)
        compute_raw[step_key] = max(ml) if any(x > 0 for x in ml) else 0.0
        machine_loads[step_key] = ml

    return dict(net_raw), dict(proc_raw), compute_raw, machine_loads


def load_or_compute_components(trace_path, partition_path, out_presence, raw_recv_by_vertex, step_keys):
    """
    Component cache is keyed by (trace_mtime, partition_mtime, version).
    """
    trace_tag = os.path.basename(os.path.dirname(trace_path))
    comp_cache_path = f"{partition_path}.{trace_tag}.comp_cache.pkl"

    trace_mtime = os.path.getmtime(trace_path)
    part_mtime = os.path.getmtime(partition_path)
    cache_key = (COMP_CACHE_VERSION, trace_mtime, part_mtime)

    if os.path.isfile(comp_cache_path):
        try:
            tc = time.perf_counter()
            with open(comp_cache_path, "rb") as fh:
                old_key, cached_result = pickle.load(fh)
            if old_key == cache_key:
                print(f"    comp cache hit ({time.perf_counter() - tc:.1f}s)")
                return cached_result
            print("    comp cache stale — recomputing...")
        except Exception as e:
            print(f"    comp cache load failed ({e}) — recomputing...")

    net_raw, proc_raw, compute_raw, machine_loads = compute_raw_components(
        out_presence=out_presence,
        raw_recv_by_vertex=raw_recv_by_vertex,
        machine_of=load_partition(partition_path)[0],
        worker_of=load_partition(partition_path)[1],
        step_keys=step_keys,
    )

    try:
        with open(comp_cache_path, "wb") as fh:
            pickle.dump(
                (cache_key, (net_raw, proc_raw, compute_raw, machine_loads)),
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"    comp cache written → {comp_cache_path}")
    except Exception as e:
        print(f"    comp cache write failed ({e})")

    return net_raw, proc_raw, compute_raw, machine_loads


# =============================================================================
# 3. Fitting / metrics
# =============================================================================

def fit_nnls_rows(rows):
    """
    rows: list[(run_idx, superstep, net, proc, compute, actual_time)]

    Model:
        time = c_bias + c_net*net + c_proc*proc + c_node*compute

    Returns
    -------
    (c_bias, c_net, c_proc, c_node, r2, rmse_pct, detail_rows)
    or None
    """
    if np is None:
        return None

    valid_rows = [r for r in rows if r[5] is not None and r[5] > 0]
    if len(valid_rows) < 4:
        return None

    X = np.array(
        [[1.0, r[2], r[3], r[4]] for r in valid_rows],
        dtype=np.float64,
    )
    y = np.array([r[5] for r in valid_rows], dtype=np.float64)

    if _HAVE_SCIPY:
        coeffs, _ = _scipy_nnls(X, y)
    else:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        coeffs = np.maximum(coeffs, 0.0)

    c_bias = float(coeffs[0])
    c_net = float(coeffs[1])
    c_proc = float(coeffs[2])
    c_node = float(coeffs[3])

    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0
    msre = float(np.mean(((y_pred - y) / y) ** 2))
    rmse_pct = 100.0 * math.sqrt(msre)

    detail_rows = [
        (r[0], r[1], r[5], float(pred))
        for r, pred in zip(valid_rows, y_pred)
    ]

    return c_bias, c_net, c_proc, c_node, r2, rmse_pct, detail_rows


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None, n

    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5

    if dx < 1e-12 or dy < 1e-12:
        return None, n

    return num / (dx * dy), n


# =============================================================================
# 4. Display
# =============================================================================

def print_step_table(
    run_label,
    step_keys,
    net_raw,
    proc_raw,
    compute_raw,
    machine_loads,
    c_net,
    c_proc,
    c_node,
    step_timings=None,
    actual_msgs=None,
):
    """
    Print proxy-weighted per-step table for one run.
    """
    nets = [c_net * net_raw.get(sk, 0.0) for sk in step_keys]
    procs = [c_proc * proc_raw.get(sk, 0.0) for sk in step_keys]
    computes = [c_node * compute_raw.get(sk, 0.0) for sk in step_keys]
    proxy_totals = [n + p + c for n, p, c in zip(nets, procs, computes)]

    have_timings = step_timings is not None
    width = 130 if have_timings else 104

    print(f"\n{'=' * width}")
    print(f"  {run_label}")
    print(f"  c_net={c_net:.2e}  c_proc={c_proc:.2e}  c_node={c_node:.2e}")
    print(f"{'=' * width}")

    if have_timings:
        print(
            f"  {'Task':<12} {'Step':>5}  "
            f"{'Net':>12} {'Proc':>12} {'Compute':>12} {'ProxyTotal':>12}  "
            f"{'Actual(s)':>10} {'Ratio':>7}  {'BotM':>4}  ActMsgs/CW/CM"
        )
        print(
            f"  {'-' * 12} {'-' * 5}  "
            f"{'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}  "
            f"{'-' * 10} {'-' * 7}  {'-' * 4}  {'-' * 30}"
        )
    else:
        print(
            f"  {'Task':<12} {'Step':>5}  "
            f"{'Net':>14} {'Proc':>14} {'Compute':>14} {'ProxyTotal':>14}  "
            f"{'BotM':>4}  MachineLoads"
        )
        print(
            f"  {'-' * 12} {'-' * 5}  "
            f"{'-' * 14} {'-' * 14} {'-' * 14} {'-' * 14}  "
            f"{'-' * 4}  {'-' * 25}"
        )

    for i, sk in enumerate(step_keys):
        task, s = sk
        ml = machine_loads.get(sk, [])
        bot_m = ml.index(max(ml)) if ml else -1
        bot_load = max(ml) if ml else 0.0
        avg_load = sum(ml) / len(ml) if ml else 0.0

        if have_timings:
            act_t = step_timings.get(s)
            ratio_str = f"{proxy_totals[i] / act_t:>7.3f}" if act_t and act_t > 0 else f"{'N/A':>7}"
            act_str = f"{act_t:>10.6f}" if act_t is not None else f"{'N/A':>10}"

            am = (actual_msgs or {}).get(s, {})
            msg_str = (
                f"{am.get('msgs', 0):,}/"
                f"{am.get('cross_worker', 0):,}/"
                f"{am.get('cross_machine', 0):,}"
            )

            print(
                f"  {task:<12} {s:>5}  "
                f"{nets[i]:>12,.1f} {procs[i]:>12,.1f} {computes[i]:>12,.1f} {proxy_totals[i]:>12,.1f}  "
                f"{act_str} {ratio_str}  m{bot_m:>3}  {msg_str}"
            )
        else:
            load_str = f"max={bot_load:,.0f}  avg={avg_load:,.0f}"
            print(
                f"  {task:<12} {s:>5}  "
                f"{nets[i]:>14,.1f} {procs[i]:>14,.1f} {computes[i]:>14,.1f} {proxy_totals[i]:>14,.1f}  "
                f"m{bot_m:>3}  {load_str}"
            )

    tn = sum(nets)
    tp = sum(procs)
    tc = sum(computes)
    tt = sum(proxy_totals)

    print(f"  {'':->12} {'':->5}  {'':->12} {'':->12} {'':->12} {'':->12}")
    if have_timings:
        act_total = sum(
            step_timings.get(sk[1], 0.0)
            for sk in step_keys
            if step_timings.get(sk[1]) is not None
        )
        print(f"  {'TOTAL':<12} {'':>5}  {tn:>12,.1f} {tp:>12,.1f} {tc:>12,.1f} {tt:>12,.1f}  {act_total:>10.4f}")
    else:
        print(f"  {'TOTAL':<12} {'':>5}  {tn:>12,.1f} {tp:>12,.1f} {tc:>12,.1f} {tt:>12,.1f}")


# =============================================================================
# 5. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Leave-one-run-out cross-validation for Pregglenator runtime fitting"
    )
    ap.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="Trace files, one per run",
    )
    ap.add_argument(
        "--timings",
        nargs="+",
        required=True,
        help="Timing JSON files, one per run",
    )
    ap.add_argument(
        "--partitions",
        nargs="+",
        required=True,
        help="Partition JSON files, one per run",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional human-readable labels, one per run",
    )
    ap.add_argument("--c_net", type=float, default=1e-7)
    ap.add_argument("--c_proc", type=float, default=1e-7)
    ap.add_argument("--c_node", type=float, default=1e-7)
    ap.add_argument(
        "--print_step_tables",
        action="store_true",
        help="Print per-step proxy tables for each run",
    )

    args = ap.parse_args()

    n_runs = len(args.traces)
    if len(args.timings) != n_runs or len(args.partitions) != n_runs:
        ap.error("--traces, --timings, and --partitions must have the same length")

    if args.labels is not None and len(args.labels) != n_runs:
        ap.error("--labels must match the number of runs")

    if np is None:
        raise RuntimeError("numpy is required for CV fitting")

    t0 = time.perf_counter()

    print("\n[1/3] Loading runs and computing features...")
    runs = []

    for i in range(n_runs):
        label = args.labels[i] if args.labels is not None else (
            f"run{i}:src={os.path.basename(os.path.dirname(args.traces[i]))},part={os.path.basename(args.partitions[i])}"
        )
        print(f"\n  Run {i + 1}/{n_runs}: {label}")

        out_presence, raw_recv_by_vertex, step_keys = load_trace(args.traces[i])
        step_timings, actual_msgs, total_time, src = load_timings(args.timings[i])
        machine_of, worker_of = load_partition(args.partitions[i])

        trace_tag = os.path.basename(os.path.dirname(args.traces[i]))
        comp_cache_path = f"{args.partitions[i]}.{trace_tag}.comp_cache.pkl"
        trace_mtime = os.path.getmtime(args.traces[i])
        part_mtime = os.path.getmtime(args.partitions[i])
        cache_key = (COMP_CACHE_VERSION, trace_mtime, part_mtime)

        comp_loaded = False
        if os.path.isfile(comp_cache_path):
            try:
                tc = time.perf_counter()
                with open(comp_cache_path, "rb") as fh:
                    old_key, cached_result = pickle.load(fh)
                if old_key == cache_key:
                    net_raw, proc_raw, compute_raw, machine_loads = cached_result
                    print(f"    comp cache hit ({time.perf_counter() - tc:.1f}s)")
                    comp_loaded = True
                else:
                    print("    comp cache stale — recomputing...")
            except Exception as e:
                print(f"    comp cache load failed ({e}) — recomputing...")

        if not comp_loaded:
            net_raw, proc_raw, compute_raw, machine_loads = compute_raw_components(
                out_presence=out_presence,
                raw_recv_by_vertex=raw_recv_by_vertex,
                machine_of=machine_of,
                worker_of=worker_of,
                step_keys=step_keys,
            )
            try:
                with open(comp_cache_path, "wb") as fh:
                    pickle.dump(
                        (cache_key, (net_raw, proc_raw, compute_raw, machine_loads)),
                        fh,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(f"    comp cache written → {comp_cache_path}")
            except Exception as e:
                print(f"    comp cache write failed ({e})")

        tn_raw = sum(net_raw.get(sk, 0.0) for sk in step_keys)
        tp_raw = sum(proc_raw.get(sk, 0.0) for sk in step_keys)
        tc_raw = sum(compute_raw.get(sk, 0.0) for sk in step_keys)

        proxy_net = args.c_net * tn_raw
        proxy_proc = args.c_proc * tp_raw
        proxy_compute = args.c_node * tc_raw
        proxy_total = proxy_net + proxy_proc + proxy_compute

        if args.print_step_tables:
            print_step_table(
                run_label=label,
                step_keys=step_keys,
                net_raw=net_raw,
                proc_raw=proc_raw,
                compute_raw=compute_raw,
                machine_loads=machine_loads,
                c_net=args.c_net,
                c_proc=args.c_proc,
                c_node=args.c_node,
                step_timings=step_timings,
                actual_msgs=actual_msgs,
            )

        runs.append({
            "label": label,
            "trace": args.traces[i],
            "timings": args.timings[i],
            "partition": args.partitions[i],
            "step_keys": step_keys,
            "step_timings": step_timings,
            "actual_msgs": actual_msgs,
            "actual_total": total_time,
            "source": src,
            "net_raw": net_raw,
            "proc_raw": proc_raw,
            "compute_raw": compute_raw,
            "machine_loads": machine_loads,
            "raw_totals": (tn_raw, tp_raw, tc_raw),
            "proxy_components": (proxy_net, proxy_proc, proxy_compute),
            "proxy_total": proxy_total,
        })

    print("\n[2/3] Leave-one-run-out CV...")
    cv_results = []

    for holdout_idx in range(n_runs):
        train_rows = []

        for j in range(n_runs):
            if j == holdout_idx:
                continue

            rj = runs[j]
            for sk in rj["step_keys"]:
                s = sk[1]
                act = rj["step_timings"].get(s)
                if act is None or act <= 0:
                    continue

                train_rows.append((
                    j,
                    s,
                    rj["net_raw"].get(sk, 0.0),
                    rj["proc_raw"].get(sk, 0.0),
                    rj["compute_raw"].get(sk, 0.0),
                    act,
                ))

        fit = fit_nnls_rows(train_rows)
        if fit is None:
            raise RuntimeError(f"Could not fit fold with holdout run {holdout_idx}")

        c_bias, c_net_fit, c_proc_fit, c_node_fit, r2_train, rmse_train, detail_rows = fit

        rtest = runs[holdout_idx]
        heldout_rows = []
        for sk in rtest["step_keys"]:
            s = sk[1]
            act = rtest["step_timings"].get(s)
            if act is None or act <= 0:
                continue
            heldout_rows.append((
                s,
                rtest["net_raw"].get(sk, 0.0),
                rtest["proc_raw"].get(sk, 0.0),
                rtest["compute_raw"].get(sk, 0.0),
                act,
            ))

        pred_step_rows = []
        pred_total = 0.0
        act_total = 0.0
        for s, nr, pr, cr, act in heldout_rows:
            pred = c_bias + c_net_fit * nr + c_proc_fit * pr + c_node_fit * cr
            pred_step_rows.append((s, act, pred))
            pred_total += pred
            act_total += act

        abs_err = pred_total - act_total
        pct_err = 100.0 * abs_err / act_total if act_total else float("inf")

        cv_results.append({
            "holdout_idx": holdout_idx,
            "label": rtest["label"],
            "actual_total": act_total,
            "pred_total": pred_total,
            "abs_err": abs_err,
            "pct_err": pct_err,
            "c_bias": c_bias,
            "c_net": c_net_fit,
            "c_proc": c_proc_fit,
            "c_node": c_node_fit,
            "train_r2": r2_train,
            "train_rmse_pct": rmse_train,
            "pred_step_rows": pred_step_rows,
        })

    print("\n[3/3] Reporting CV results...")

    print("\nPer-run totals:")
    print(
        f"  {'Run':<35} {'ProxyTotal':>12} {'Actual':>10} {'CV Pred':>10} "
        f"{'Proxy/Act':>10} {'Pred/Act':>10} {'CV Err%':>9}"
    )
    print(
        f"  {'-' * 35} {'-' * 12} {'-' * 10} {'-' * 10} "
        f"{'-' * 10} {'-' * 10} {'-' * 9}"
    )

    proxy_totals = []
    actual_totals = []
    pred_totals = []

    for run, cv in zip(runs, cv_results):
        proxy = run["proxy_total"]
        act = cv["actual_total"]
        pred = cv["pred_total"]

        proxy_totals.append(proxy)
        actual_totals.append(act)
        pred_totals.append(pred)

        proxy_ratio = proxy / act if act else float("inf")
        pred_ratio = pred / act if act else float("inf")

        print(
            f"  {cv['label']:<35} {proxy:>12,.4f} {act:>10.4f} {pred:>10.4f} "
            f"{proxy_ratio:>10.4f} {pred_ratio:>10.4f} {cv['pct_err']:>+8.2f}%"
        )

    print("\nFold coefficients:")
    print(
        f"  {'Holdout':<35} {'c_bias':>10} {'c_net':>10} {'c_proc':>10} {'c_node':>10} "
        f"{'Train R²':>10} {'Train RMSE%':>12}"
    )
    print(
        f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} "
        f"{'-' * 10} {'-' * 12}"
    )
    for cv in cv_results:
        print(
            f"  {cv['label']:<35} "
            f"{cv['c_bias']:>10.2e} {cv['c_net']:>10.2e} {cv['c_proc']:>10.2e} {cv['c_node']:>10.2e} "
            f"{cv['train_r2']:>10.4f} {cv['train_rmse_pct']:>12.2f}"
        )

    mean_abs_pct = sum(abs(cv["pct_err"]) for cv in cv_results) / len(cv_results)
    rmse_pct = 100.0 * math.sqrt(
        sum(((cv["pred_total"] - cv["actual_total"]) / cv["actual_total"]) ** 2 for cv in cv_results if cv["actual_total"] > 0)
        / max(1, sum(1 for cv in cv_results if cv["actual_total"] > 0))
    )

    pred_r, _ = _pearson(pred_totals, actual_totals)
    proxy_r, _ = _pearson(proxy_totals, actual_totals)

    pred_r_str = f"{pred_r:.4f}" if pred_r is not None else "N/A"
    proxy_r_str = f"{proxy_r:.4f}" if proxy_r is not None else "N/A"

    print("\nAggregate CV metrics:")
    print(f"  Mean absolute held-out % error : {mean_abs_pct:.2f}%")
    print(f"  Held-out RMSE%                 : {rmse_pct:.2f}%")
    print(f"  Pearson r(pred, actual)        : {pred_r_str}")
    print(f"  Pearson r(proxy, actual)       : {proxy_r_str}")

    print("\nAverage fitted coefficients across folds:")
    print(f"  c_bias = {sum(cv['c_bias'] for cv in cv_results) / len(cv_results):.4e}")
    print(f"  c_net  = {sum(cv['c_net'] for cv in cv_results) / len(cv_results):.4e}")
    print(f"  c_proc = {sum(cv['c_proc'] for cv in cv_results) / len(cv_results):.4e}")
    print(f"  c_node = {sum(cv['c_node'] for cv in cv_results) / len(cv_results):.4e}")

    print(f"\n[done] total elapsed: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()