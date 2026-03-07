#!/usr/bin/env python3
"""
pregglenator_verify.py

Loads a single Pregel merged.csv trace plus one or more partition JSONs and:

1. Computes combiner-aware raw cost components per superstep
2. Prints a per-step proxy-cost breakdown for each partition
3. Compares partitions side-by-side under the proxy objective
4. Optionally compares proxy totals against observed actual runtimes
5. If per-step timings are available, fits a nonnegative linear runtime model:
       time(step) = c_bias + c_net*net + c_proc*proc + c_node*compute
   using partition[0] as the calibration partition
6. Applies that fitted model to all partitions and reports ranking metrics

Notes
-----
- "Proxy" totals are the user-supplied objective:
      c_net*net + c_proc*proc + c_node*compute
  and are unitless unless you intentionally chose time-calibrated weights.

- "Fitted" totals are NNLS-predicted seconds:
      S*c_bias + c_net*net_total + c_proc*proc_total + c_node*compute_total
  where S is the number of active timed supersteps used for calibration.

- The NNLS model is fit on partition[0] only, then transferred to all partitions.
  That makes the fitted totals a calibration-transfer estimate, not an independent
  per-partition fit.
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


# =============================================================================
# 1. Trace loading
# =============================================================================

def load_trace(trace_path):
    """Load a single merged.csv trace file, using a pickle cache when available."""
    if not os.path.isfile(trace_path):
        raise RuntimeError(f"Trace file not found: {trace_path}")

    task_name = os.path.basename(os.path.dirname(trace_path))
    print(f"  task: {task_name}  ({trace_path})")

    cache_path = trace_path + ".verify.pkl"
    trace_mtime = os.path.getmtime(trace_path)

    if os.path.isfile(cache_path):
        try:
            t_load = time.perf_counter()
            with open(cache_path, "rb") as fh:
                cached_mtime, result = pickle.load(fh)
            if cached_mtime == trace_mtime:
                out_presence, raw_recv_by_vertex, step_keys = result
                print(
                    f"  cache hit  ({time.perf_counter() - t_load:.1f}s)  |  "
                    f"{len(step_keys)} task-steps  |  {len(out_presence):,} source vertices"
                )
                return out_presence, raw_recv_by_vertex, step_keys
            print("  cache stale (trace modified) — rebuilding...")
        except Exception as e:
            print(f"  cache load failed ({e}) — rebuilding...")

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
        print(f"  pandas: {len(df):,} rows -> {len(grouped):,} unique (s,u,v) pairs")

        recv_grouped = df.groupby(
            ["superstep", "dst_vertex"],
            sort=False
        )["count"].sum()

        for (s, v), csum in tqdm(
            recv_grouped.items(),
            total=len(recv_grouped),
            desc="  Building recv index",
            unit="pair",
            mininterval=0.5,
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
            desc="  Building presence index",
            unit="edge",
            mininterval=0.5,
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

            for row in tqdm(reader, desc="Reading trace", unit="row"):
                s = int(row[si])
                u = int(row[ui])
                v = int(row[vi])
                c = float(row[ci])
                if u != v and c > 0:
                    local_counts[(s, u, v)] += c

        for (s, u, v), csum in tqdm(
            local_counts.items(),
            total=len(local_counts),
            desc="  Building index",
            unit="pair",
            mininterval=0.5,
        ):
            step_key = (task_name, int(s))
            step_keys_set.add(step_key)
            out_presence[int(u)].append((step_key, int(v)))
            raw_recv_by_vertex[int(v)][step_key] += float(csum)

    step_keys = sorted(step_keys_set, key=lambda x: (x[0], x[1]))
    out_presence = dict(out_presence)
    raw_recv_by_vertex = {v: dict(ss) for v, ss in raw_recv_by_vertex.items()}

    try:
        t_write = time.perf_counter()
        with open(cache_path, "wb") as fh:
            pickle.dump(
                (trace_mtime, (out_presence, raw_recv_by_vertex, step_keys)),
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"  cache written to {cache_path}  ({time.perf_counter() - t_write:.1f}s)")
    except Exception as e:
        print(f"  cache write failed ({e}) — continuing without cache")

    return out_presence, raw_recv_by_vertex, step_keys


# =============================================================================
# 2. Partition loading
# =============================================================================

def load_partition(path):
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
    Load a timing JSON file (e.g. eval_src_9783.json).

    Returns
    -------
    times : {superstep: elapsed_seconds}
    actual_msgs : {superstep: {"msgs", "cross_worker", "cross_machine"}}
    total_time : float | None
    source : Any
    """
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
# 3. Cost computation
# =============================================================================

def compute_raw_components(out_presence, raw_recv_by_vertex, machine_of, worker_of, step_keys):
    """
    Precompute the three unscaled cost components per task-step.

    Returns
    -------
    net_raw : {step_key: float}
        Distinct cross-machine sender-machine events.
    proc_raw : {step_key: float}
        Distinct cross-worker same-machine sender-worker events.
    compute_raw : {step_key: float}
        Max machine recv load in that step.
    machine_loads : {step_key: list[float]}
        Per-machine recv load vector in that step.
    """
    num_machines = (max(machine_of.values()) + 1) if machine_of else 1

    machine_hist = defaultdict(set)   # (dst, step_key) -> {sender_machine, ...}
    worker_hist = defaultdict(set)    # (dst, step_key) -> {sender_worker, ...}

    total_events = sum(len(v) for v in out_presence.values())

    with tqdm(
        total=len(out_presence),
        desc="  Sender histograms",
        unit="vertex",
        mininterval=0.5,
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
        f"  histograms: {len(machine_hist):,} cross-machine (dst,step) pairs  |  "
        f"{len(worker_hist):,} cross-worker pairs  |  {total_events:,} total presence events"
    )

    net_raw = defaultdict(float)
    proc_raw = defaultdict(float)

    for (dst, step_key), hist in tqdm(
        machine_hist.items(),
        desc="  Net cost aggregation",
        unit="pair",
        mininterval=0.5,
    ):
        net_raw[step_key] += len(hist)

    for (dst, step_key), hist in tqdm(
        worker_hist.items(),
        desc="  Proc cost aggregation",
        unit="pair",
        mininterval=0.5,
    ):
        wd = worker_of.get(dst)
        proc_raw[step_key] += len(hist) - (1 if wd in hist else 0)

    step_machine_loads = defaultdict(lambda: [0.0] * num_machines)
    for v, ss in tqdm(
        raw_recv_by_vertex.items(),
        desc="  Compute bottleneck",
        unit="vertex",
        mininterval=0.5,
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


# =============================================================================
# 4. Display helpers
# =============================================================================

def _pearson(xs, ys):
    """Pearson correlation, returns (r, n) or (None, 0) if not computable."""
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


def _average_ranks(values):
    """
    Average-rank handling for ties.
    Returns ranks in [0, n-1] on average-rank scale.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


def _spearman_tie_aware(xs, ys):
    """
    Tie-aware Spearman via Pearson on average ranks.
    Returns (rho, n) or (None, 0).
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None, 0

    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    return _pearson(rx, ry)


def print_step_table(
    step_keys,
    net_raw,
    proc_raw,
    compute_raw,
    machine_loads,
    c_net,
    c_proc,
    c_node,
    partition_label,
    num_machines,
    step_timings=None,
    actual_msgs=None,
):
    """
    Print proxy-weighted per-step table.

    step_timings : {superstep_int: elapsed_seconds} or None
    actual_msgs  : {superstep_int: {msgs, cross_worker, cross_machine}} or None
    """
    nets = [c_net * net_raw.get(sk, 0.0) for sk in step_keys]
    procs = [c_proc * proc_raw.get(sk, 0.0) for sk in step_keys]
    computes = [c_node * compute_raw.get(sk, 0.0) for sk in step_keys]
    proxy_totals = [n + p + c for n, p, c in zip(nets, procs, computes)]

    have_timings = step_timings is not None
    width = 130 if have_timings else 104

    print(f"\n{'=' * width}")
    print(f"  {partition_label}")
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

    est_list = []
    actual_list = []

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

            if act_t is not None:
                est_list.append(proxy_totals[i])
                actual_list.append(act_t)
        else:
            load_str = f"max={bot_load:,.0f}  avg={avg_load:,.0f}"
            print(
                f"  {task:<12} {s:>5}  "
                f"{nets[i]:>14,.1f} {procs[i]:>14,.1f} {computes[i]:>14,.1f} {proxy_totals[i]:>14,.1f}  "
                f"m{bot_m:>3}  {load_str}"
            )

    print(f"  {'':->12} {'':->5}  {'':->12} {'':->12} {'':->12} {'':->12}")
    tn = sum(nets)
    tp = sum(procs)
    tc = sum(computes)
    tt = sum(proxy_totals)

    if have_timings:
        act_total = sum(
            step_timings.get(sk[1], 0.0)
            for sk in step_keys
            if step_timings.get(sk[1]) is not None
        )
        print(f"  {'TOTAL':<12} {'':>5}  {tn:>12,.1f} {tp:>12,.1f} {tc:>12,.1f} {tt:>12,.1f}  {act_total:>10.4f}")
    else:
        print(f"  {'TOTAL':<12} {'':>5}  {tn:>14,.1f} {tp:>14,.1f} {tc:>14,.1f} {tt:>14,.1f}")

    if have_timings and len(est_list) >= 2:
        r, n = _pearson(est_list, actual_list)
        r_str = f"{r:.4f}" if r is not None else "N/A"
        print(f"\n  Pearson r (proxy estimate vs actual time, {n} supersteps): {r_str}")

    print()
    print("  Raw (unscaled) totals:")
    print(
        f"    net_raw     = {sum(net_raw.get(sk, 0.0) for sk in step_keys):>12,.0f}"
        f"   (distinct cross-machine sender-machine events)"
    )
    print(
        f"    proc_raw    = {sum(proc_raw.get(sk, 0.0) for sk in step_keys):>12,.0f}"
        f"   (distinct cross-worker same-machine sender-worker events)"
    )
    print(
        f"    compute_raw = {sum(compute_raw.get(sk, 0.0) for sk in step_keys):>12,.1f}"
        f"   (sum of per-step max-machine recv loads)"
    )


def print_comparison_table(partition_labels, all_proxy_components, fitted_totals=None):
    """
    Side-by-side total comparison across partitions.

    all_proxy_components : list of (net_total, proc_total, compute_total)
        These are proxy-weighted totals under user-supplied weights.
    fitted_totals : optional list[float]
        NNLS-fitted predicted runtimes in seconds.
    """
    has_fitted = fitted_totals is not None
    width = 112 if has_fitted else 92

    print(f"\n{'=' * width}")
    print("  PARTITION COMPARISON")
    print(f"{'=' * width}")

    if has_fitted:
        print(
            f"  {'Partition':<40}  {'Net':>14} {'Proc':>12} {'Compute':>12} "
            f"{'ProxyTotal':>12}  {'FittedTotal':>12}"
        )
        print(
            f"  {'-' * 40}  {'-' * 14} {'-' * 12} {'-' * 12} "
            f"{'-' * 12}  {'-' * 12}"
        )
    else:
        print(
            f"  {'Partition':<40}  {'Net':>14} {'Proc':>12} {'Compute':>12} {'ProxyTotal':>12}"
        )
        print(
            f"  {'-' * 40}  {'-' * 14} {'-' * 12} {'-' * 12} {'-' * 12}"
        )

    proxy_totals = [t[0] + t[1] + t[2] for t in all_proxy_components]
    best_proxy = min(proxy_totals) if proxy_totals else 0.0
    best_fitted = min(fitted_totals) if has_fitted else None

    for i, (label, (tn, tp, tc)) in enumerate(zip(partition_labels, all_proxy_components)):
        short = os.path.basename(label)
        pt = proxy_totals[i]
        pm = " *" if abs(pt - best_proxy) < 1e-12 else "  "

        if has_fitted:
            ft = fitted_totals[i]
            fm = " *" if abs(ft - best_fitted) < 1e-12 else "  "
            print(
                f"  {short:<40}  {tn:>14,.1f} {tp:>12,.1f} {tc:>12,.1f} "
                f"{pt:>12,.1f}{pm}  {ft:>12.4f}{fm}"
            )
        else:
            print(
                f"  {short:<40}  {tn:>14,.1f} {tp:>12,.1f} {tc:>12,.1f} {pt:>12,.1f}{pm}"
            )

    print("\n  * = lowest in column")
    if has_fitted:
        print("  ProxyTotal  = c_net*net + c_proc*proc + c_node*compute  (user-supplied weights, unitless)")
        print("  FittedTotal = S*c_bias + fitted_net + fitted_proc + fitted_compute  (seconds, transferred NNLS model)")


# =============================================================================
# 5. Linear fit
# =============================================================================

def fit_nnls(net_raw, proc_raw, compute_raw, step_timings, step_keys):
    """
    Fit c_bias, c_net, c_proc, c_node by non-negative least squares.

    Model:
        time(step) = c_bias + c_net*net + c_proc*proc + c_node*compute

    Returns
    -------
    (c_bias, c_net, c_proc, c_node, r2, rmse_pct, detail_rows)
    where detail_rows = [(superstep_id, actual, predicted), ...], sorted by superstep.

    Returns None if numpy is unavailable or too few data points.
    """
    if np is None:
        return None

    active_sks = []
    for sk in step_keys:
        act = step_timings.get(sk[1])
        if act is None or act <= 0:
            continue
        active_sks.append((sk, act))

    if len(active_sks) < 4:
        return None

    X = np.array(
        [
            [
                1.0,
                net_raw.get(sk, 0.0),
                proc_raw.get(sk, 0.0),
                compute_raw.get(sk, 0.0),
            ]
            for sk, _ in active_sks
        ],
        dtype=np.float64,
    )
    y = np.array([act for _, act in active_sks], dtype=np.float64)

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

    detail_rows = sorted(
        [(sk[1], act, float(yp)) for (sk, act), yp in zip(active_sks, y_pred)],
        key=lambda r: r[0],
    )

    return c_bias, c_net, c_proc, c_node, r2, rmse_pct, detail_rows


# =============================================================================
# 6. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Verify pregglenator cost estimates against trace data"
    )
    ap.add_argument(
        "--trace",
        default="../comm_traces/src_9783/merged.csv",
        help="Single merged.csv trace file to load",
    )
    ap.add_argument(
        "--timings",
        default="eval_src_9783.json",
        help="Timing JSON file (eval_src_*.json) for per-superstep actual runtimes",
    )
    ap.add_argument(
        "--partitions",
        nargs="+",
        default=["random.json"],
        help="One or more partition JSON files to analyse",
    )
    ap.add_argument("--c_net", type=float, default=1e-7)
    ap.add_argument("--c_proc", type=float, default=1e-7)
    ap.add_argument("--c_node", type=float, default=1e-7)
    ap.add_argument(
        "--actual_runtimes",
        nargs="+",
        type=float,
        default=None,
        help="Observed runtimes (one per --partitions entry, same order). Used for ranking metrics.",
    )
    ap.add_argument(
        "--fit_partition_idx",
        type=int,
        default=0,
        help="Which partition index to use for NNLS calibration if timings are available.",
    )

    args = ap.parse_args()

    if args.actual_runtimes is not None and len(args.actual_runtimes) != len(args.partitions):
        ap.error("--actual_runtimes must have the same number of entries as --partitions")

    if not (0 <= args.fit_partition_idx < len(args.partitions)):
        ap.error("--fit_partition_idx must be in [0, len(--partitions)-1]")

    t0 = time.perf_counter()

    print("\n[1/4] Loading trace...")
    t1 = time.perf_counter()
    out_presence, raw_recv_by_vertex, step_keys = load_trace(args.trace)
    print(
        f"  done in {time.perf_counter() - t1:.2f}s  |  "
        f"{len(step_keys)} task-steps  |  {len(out_presence):,} active source vertices  |  "
        f"{sum(len(v) for v in out_presence.values()):,} total presence events"
    )

    print("\n[2/4] Loading timings...")
    step_timings = None
    actual_msgs = None

    if args.timings and os.path.isfile(args.timings):
        step_timings, actual_msgs, total_time, src = load_timings(args.timings)
        print(f"  src={src}  total={total_time}s  {len(step_timings)} supersteps loaded")
    elif args.timings:
        print(f"  Warning: timings file not found: {args.timings}")
    else:
        print("  (no timings file)")

    n_parts = len(args.partitions)
    all_labels = []
    all_raw_totals = []
    all_proxy_components = []
    all_per_step_raw = []

    for part_idx, partition_path in enumerate(args.partitions):
        print(f"\n[3/4] Partition {part_idx + 1}/{n_parts}: {partition_path}")
        t2 = time.perf_counter()
        machine_of, worker_of = load_partition(partition_path)
        num_machines = (max(machine_of.values()) + 1) if machine_of else 1
        print(
            f"  loaded in {time.perf_counter() - t2:.2f}s  |  "
            f"{num_machines} machines  |  {len(machine_of):,} nodes assigned"
        )

        comp_cache_path = partition_path + ".comp_cache.pkl"
        trace_mtime = os.path.getmtime(args.trace) if os.path.isfile(args.trace) else 0
        part_mtime = os.path.getmtime(partition_path)
        cache_key = (trace_mtime, part_mtime)
        comp_loaded = False

        if os.path.isfile(comp_cache_path):
            try:
                tc = time.perf_counter()
                with open(comp_cache_path, "rb") as fh:
                    old_key, cached_result = pickle.load(fh)
                if old_key == cache_key:
                    net_raw, proc_raw, compute_raw, machine_loads = cached_result
                    print(f"  comp cache hit  ({time.perf_counter() - tc:.1f}s)  →  {comp_cache_path}")
                    comp_loaded = True
                else:
                    print("  comp cache stale — recomputing...")
            except Exception as e:
                print(f"  comp cache load failed ({e}) — recomputing...")

        if not comp_loaded:
            print("  Computing raw cost components...")
            t3 = time.perf_counter()
            net_raw, proc_raw, compute_raw, machine_loads = compute_raw_components(
                out_presence,
                raw_recv_by_vertex,
                machine_of,
                worker_of,
                step_keys,
            )
            print(f"  raw components done in {time.perf_counter() - t3:.2f}s")
            try:
                with open(comp_cache_path, "wb") as fh:
                    pickle.dump(
                        (cache_key, (net_raw, proc_raw, compute_raw, machine_loads)),
                        fh,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                print(f"  comp cache written → {comp_cache_path}")
            except Exception as e:
                print(f"  comp cache write failed ({e})")

        print("  Printing step table...")
        print_step_table(
            step_keys=step_keys,
            net_raw=net_raw,
            proc_raw=proc_raw,
            compute_raw=compute_raw,
            machine_loads=machine_loads,
            c_net=args.c_net,
            c_proc=args.c_proc,
            c_node=args.c_node,
            partition_label=partition_path,
            num_machines=num_machines,
            step_timings=step_timings,
            actual_msgs=actual_msgs,
        )

        tn_raw = sum(net_raw.get(sk, 0.0) for sk in step_keys)
        tp_raw = sum(proc_raw.get(sk, 0.0) for sk in step_keys)
        tc_raw = sum(compute_raw.get(sk, 0.0) for sk in step_keys)

        all_labels.append(partition_path)
        all_raw_totals.append((tn_raw, tp_raw, tc_raw))
        all_proxy_components.append((
            args.c_net * tn_raw,
            args.c_proc * tp_raw,
            args.c_node * tc_raw,
        ))
        all_per_step_raw.append((net_raw, proc_raw, compute_raw))

    nnls_result = None
    nnls_fitted_totals = None

    if step_timings is not None and all_per_step_raw:
        fit_idx = args.fit_partition_idx
        nr, pr, cr = all_per_step_raw[fit_idx]
        nnls_result = fit_nnls(nr, pr, cr, step_timings, step_keys)

    if nnls_result is not None:
        fb, fn, fp, fc, r2, rmse_pct, detail_rows = nnls_result
        n_active = len(detail_rows)

        nnls_fitted_totals = [
            n_active * fb + fn * tn_r + fp * tp_r + fc * tc_r
            for tn_r, tp_r, tc_r in all_raw_totals
        ]

        solver = "scipy NNLS" if _HAVE_SCIPY else "numpy lstsq + clip fallback"

        print(f"\n  NNLS fit ({n_active} supersteps, {solver}):")
        print(f"    calibration partition index = {args.fit_partition_idx}  ({os.path.basename(all_labels[args.fit_partition_idx])})")
        print(f"    c_bias = {fb:.4e}   (per-step BSP overhead, {n_active} steps × {fb:.4e} = {n_active * fb:.4f}s)")
        print(f"    c_net  = {fn:.4e}   c_proc = {fp:.4e}   c_node = {fc:.4e}")
        print(f"    R²     = {r2:.4f}    RMSE%  = {rmse_pct:.2f}%")
        print(f"    Note: fitted coefficients are calibrated on partition[{args.fit_partition_idx}] and transferred to all partitions.")

        total_act = sum(r[1] for r in detail_rows)
        total_pred = sum(r[2] for r in detail_rows)

        print(f"\n    {'SS':>4}  {'Actual':>10}  {'Predicted':>10}  {'Err%':>7}")
        print(f"    {'-' * 4}  {'-' * 10}  {'-' * 10}  {'-' * 7}")
        for ss, act, pred in detail_rows:
            err_pct = 100.0 * (pred - act) / act if act != 0 else float("inf")
            print(f"    {ss:>4}  {act:>10.4f}  {pred:>10.4f}  {err_pct:>+7.1f}%")

        total_err_pct = 100.0 * (total_pred - total_act) / total_act if total_act != 0 else float("inf")
        print(f"    {'TOT':>4}  {total_act:>10.4f}  {total_pred:>10.4f}  {total_err_pct:>+7.1f}%")

        print("\n    To re-run pregglenator with fitted weights:")
        print(
            f"    python pregglenator_maximum_boogaloo.py "
            f"--c_net {fn:.2e} --c_proc {fp:.2e} --c_node {fc:.2e}"
        )
    elif step_timings is not None:
        print("\n  (NNLS fit skipped: numpy unavailable or too few data points)")

    if len(args.partitions) > 1:
        print_comparison_table(all_labels, all_proxy_components, nnls_fitted_totals)

    if args.actual_runtimes is not None:
        acts = args.actual_runtimes
        has_fitted = nnls_fitted_totals is not None

        print(f"\n  Estimated vs Actual  ({'proxy + fitted' if has_fitted else 'proxy only'}):")

        if has_fitted:
            print(
                f"  {'Partition':<40}  {'ProxyTotal':>12} {'Actual':>10} {'ProxyRatio':>11}  "
                f"{'FittedTotal':>12} {'FittedRatio':>12}"
            )
            print(
                f"  {'-' * 40}  {'-' * 12} {'-' * 10} {'-' * 11}  {'-' * 12} {'-' * 12}"
            )
        else:
            print(
                f"  {'Partition':<40}  {'ProxyTotal':>12} {'Actual':>10} {'ProxyRatio':>11}"
            )
            print(
                f"  {'-' * 40}  {'-' * 12} {'-' * 10} {'-' * 11}"
            )

        for i, (label, (tn, tp, tc), actual) in enumerate(zip(all_labels, all_proxy_components, acts)):
            proxy = tn + tp + tc
            proxy_ratio = proxy / actual if actual else float("inf")
            row = f"  {os.path.basename(label):<40}  {proxy:>12,.1f} {actual:>10.3f} {proxy_ratio:>11.4f}"

            if has_fitted:
                ft = nnls_fitted_totals[i]
                fitted_ratio = ft / actual if actual else float("inf")
                row += f"  {ft:>12.4f} {fitted_ratio:>12.4f}"

            print(row)

        if has_fitted:
            print("\n  ProxyTotal  : unitless proxy objective using user-supplied c_net/c_proc/c_node")
            print("  FittedTotal : seconds predicted by transferred NNLS model")

    if args.actual_runtimes is not None and len(args.partitions) > 1:
        acts = args.actual_runtimes

        def _ranking_metrics(scores, label):
            n = len(scores)

            n_pairs = 0
            n_correct = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if acts[i] == acts[j]:
                        continue
                    n_pairs += 1
                    if (scores[i] < scores[j]) == (acts[i] < acts[j]):
                        n_correct += 1

            pairwise_acc = 100.0 * n_correct / n_pairs if n_pairs else float("nan")
            rho, _ = _spearman_tie_aware(scores, acts)
            rho_str = f"{rho:+.4f}" if rho is not None else "N/A"

            print(f"    {label}")
            print(f"      Spearman ρ        = {rho_str}")
            print(f"      Pairwise accuracy = {pairwise_acc:.1f}%  ({n_correct}/{n_pairs} pairs correct)")

        print(f"\n  Ranking metrics ({len(acts)} partitions vs actual runtimes):")

        proxy_scores = [tn + tp + tc for tn, tp, tc in all_proxy_components]
        _ranking_metrics(
            proxy_scores,
            "Proxy weights   (user-supplied c_net/c_proc/c_node, unitless)"
        )

        if nnls_fitted_totals is not None:
            _ranking_metrics(
                nnls_fitted_totals,
                f"NNLS transferred (fit on partition[{args.fit_partition_idx}], scored in seconds)"
            )

    print(f"\n[done] total elapsed: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()