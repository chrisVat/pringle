#!/usr/bin/env python3
"""
pregglenator_verify.py

Loads comm traces + one or more partition JSONs and computes a detailed
per-step cost breakdown using the combiner-aware cost model.

Use cases
---------
1. Inspect estimated cost breakdown for a single partition:
     python pregglenator_verify.py --trace ../comm_traces/src_0/merged.csv --partitions part.json

2. Compare estimated costs across multiple partitions side-by-side:
     python pregglenator_verify.py --trace ../comm_traces/src_0/merged.csv \
         --partitions part_a.json part_b.json part_c.json

3. Grid-search c_net / c_proc / c_node to best fit observed runtimes
   (one runtime value per partition, in the same order):
     python pregglenator_verify.py --trace ../comm_traces/src_0/merged.csv \
         --partitions part_a.json part_b.json \
         --actual_runtimes 42.1 38.7 \
         --grid_search

The grid search precomputes the three raw component totals per partition
(net_raw, proc_raw, compute_raw) and then sweeps over all (c_net, c_proc, c_node)
combinations, making it cheap regardless of grid density.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from itertools import product

from tqdm import tqdm


# =============================================================================
# 1. Trace loading
# =============================================================================

def load_trace(trace_path):
    """Load a single merged.csv trace file."""
    if not os.path.isfile(trace_path):
        raise RuntimeError(f"Trace file not found: {trace_path}")

    task_name = os.path.basename(os.path.dirname(trace_path))
    print(f"  task: {task_name}  ({trace_path})")

    local_counts = defaultdict(float)
    with open(trace_path, newline="") as fh:
        for row in tqdm(csv.DictReader(fh), desc="Reading trace", unit="row"):
            s = int(row["superstep"])
            u = int(row["src_vertex"])
            v = int(row["dst_vertex"])
            c = float(row["count"])
            if u == v or c <= 0:
                continue
            local_counts[(s, u, v)] += c

    out_presence = defaultdict(list)
    raw_recv_by_vertex = defaultdict(lambda: defaultdict(float))
    step_keys_set = set()

    for (s, u, v), csum in local_counts.items():
        step_key = (task_name, s)
        step_keys_set.add(step_key)
        out_presence[u].append((step_key, v))
        raw_recv_by_vertex[v][step_key] += csum

    step_keys = sorted(step_keys_set, key=lambda x: (x[0], x[1]))
    return (
        dict(out_presence),
        {v: dict(ss) for v, ss in raw_recv_by_vertex.items()},
        step_keys,
    )


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


# =============================================================================
# 3. Cost computation
# =============================================================================

def compute_raw_components(out_presence, raw_recv_by_vertex, machine_of, worker_of, step_keys):
    """
    Precompute the three unscaled cost components per task-step.

    Returns
    -------
    net_raw     : {step_key: float}  — distinct cross-machine sender events
    proc_raw    : {step_key: float}  — distinct cross-worker same-machine events
    compute_raw : {step_key: float}  — max machine recv load
    machine_loads : {step_key: list[float]}  — per-machine recv load vector
    """
    num_machines = (max(machine_of.values()) + 1) if machine_of else 1

    # Histograms: (dst, step_key) -> {sender_part: vertex_count}
    machine_hist = defaultdict(lambda: defaultdict(int))
    worker_hist  = defaultdict(lambda: defaultdict(int))

    for u, events in out_presence.items():
        mu = machine_of.get(u)
        wu = worker_of.get(u)
        if mu is None:
            continue
        for step_key, dst in events:
            md = machine_of.get(dst)
            if md is None:
                continue
            if mu != md:
                machine_hist[(dst, step_key)][mu] += 1
            else:
                worker_hist[(dst, step_key)][wu] += 1

    net_raw     = defaultdict(float)
    proc_raw    = defaultdict(float)

    for (dst, step_key), hist in machine_hist.items():
        net_raw[step_key] += len(hist)

    for (dst, step_key), hist in worker_hist.items():
        wd = worker_of.get(dst)
        proc_raw[step_key] += len(hist) - (1 if wd in hist else 0)

    compute_raw   = {}
    machine_loads = {}
    for step_key in step_keys:
        ml = [0.0] * num_machines
        for v, ss in raw_recv_by_vertex.items():
            if step_key in ss:
                m = machine_of.get(v)
                if m is not None:
                    ml[m] += ss[step_key]
        compute_raw[step_key]   = max(ml) if any(x > 0 for x in ml) else 0.0
        machine_loads[step_key] = ml

    return (
        dict(net_raw),
        dict(proc_raw),
        compute_raw,
        machine_loads,
    )


def scale_components(net_raw, proc_raw, compute_raw, step_keys, c_net, c_proc, c_node):
    """Apply cost weights to raw components."""
    costs = {}
    for sk in step_keys:
        costs[sk] = {
            "net":     c_net  * net_raw.get(sk, 0.0),
            "proc":    c_proc * proc_raw.get(sk, 0.0),
            "compute": c_node * compute_raw.get(sk, 0.0),
        }
    return costs


# =============================================================================
# 4. Display
# =============================================================================

def _col_width(label, values, fmt="{:,.1f}"):
    return max(len(label), max((len(fmt.format(v)) for v in values), default=0))


def print_step_table(step_keys, net_raw, proc_raw, compute_raw, machine_loads,
                     c_net, c_proc, c_node, partition_label, num_machines):
    nets     = [c_net  * net_raw.get(sk, 0.0)     for sk in step_keys]
    procs    = [c_proc * proc_raw.get(sk, 0.0)     for sk in step_keys]
    computes = [c_node * compute_raw.get(sk, 0.0)  for sk in step_keys]
    totals   = [n + p + c for n, p, c in zip(nets, procs, computes)]

    W = 100
    print(f"\n{'='*W}")
    print(f"  {partition_label}")
    print(f"  c_net={c_net:.2e}  c_proc={c_proc:.2e}  c_node={c_node:.2e}")
    print(f"{'='*W}")
    print(f"  {'Task':<12} {'Step':>5}  {'Net':>14} {'Proc':>14} {'Compute':>14} {'Total':>14}  {'BotMachine':>10}  {'Load/M'}")
    print(f"  {'-'*12} {'-'*5}  {'-'*14} {'-'*14} {'-'*14} {'-'*14}  {'-'*10}  {'-'*20}")

    for i, sk in enumerate(step_keys):
        task, s = sk
        ml = machine_loads.get(sk, [])
        bot_m   = ml.index(max(ml)) if ml else -1
        bot_load = max(ml) if ml else 0.0
        avg_load = sum(ml) / len(ml) if ml else 0.0
        load_str = f"max={bot_load:,.0f}  avg={avg_load:,.0f}"
        print(
            f"  {task:<12} {s:>5}  "
            f"{nets[i]:>14,.1f} {procs[i]:>14,.1f} {computes[i]:>14,.1f} {totals[i]:>14,.1f}  "
            f"  m{bot_m:>3}       {load_str}"
        )

    print(f"  {'':->12} {'':->5}  {'':->14} {'':->14} {'':->14} {'':->14}")
    tn, tp, tc, tt = sum(nets), sum(procs), sum(computes), sum(totals)
    print(
        f"  {'TOTAL':<12} {'':>5}  "
        f"{tn:>14,.1f} {tp:>14,.1f} {tc:>14,.1f} {tt:>14,.1f}"
    )

    # Raw component totals (useful for understanding cost model sensitivity)
    print()
    print(f"  Raw (unscaled) totals:")
    print(f"    net_raw     = {sum(net_raw.get(sk, 0.0) for sk in step_keys):>12,.0f}"
          f"   (distinct cross-machine sender-part events)")
    print(f"    proc_raw    = {sum(proc_raw.get(sk, 0.0) for sk in step_keys):>12,.0f}"
          f"   (distinct cross-worker same-machine sender-part events)")
    print(f"    compute_raw = {sum(compute_raw.get(sk, 0.0) for sk in step_keys):>12,.1f}"
          f"   (sum of per-step max-machine recv loads)")


def print_comparison_table(partition_labels, all_totals):
    """
    Side-by-side total cost comparison across partitions.
    all_totals: list of (net_total, proc_total, compute_total) per partition.
    """
    W = 90
    print(f"\n{'='*W}")
    print("  PARTITION COMPARISON  (same cost params)")
    print(f"{'='*W}")
    header = f"  {'Partition':<40}  {'Net':>14} {'Proc':>12} {'Compute':>12} {'Total':>14}"
    print(header)
    print(f"  {'-'*40}  {'-'*14} {'-'*12} {'-'*12} {'-'*14}")

    best_total = min(t[0] + t[1] + t[2] for t in all_totals) if all_totals else 0.0

    for label, (tn, tp, tc) in zip(partition_labels, all_totals):
        tt = tn + tp + tc
        marker = " *" if abs(tt - best_total) < 1e-6 else "  "
        short = os.path.basename(label)
        print(f"  {short:<40}  {tn:>14,.1f} {tp:>12,.1f} {tc:>12,.1f} {tt:>14,.1f}{marker}")

    print(f"\n  * = lowest estimated total cost")


# =============================================================================
# 5. Grid search
# =============================================================================

def grid_search(partition_raw_totals, actual_runtimes, c_net_vals, c_proc_vals, c_node_vals):
    """
    Find (c_net, c_proc, c_node) that best predicts actual_runtimes given
    precomputed (net_raw_total, proc_raw_total, compute_raw_total) per partition.

    Objective: minimise mean squared relative error across partitions, i.e.
        MSE( (estimated_i - actual_i) / actual_i )

    Parameters
    ----------
    partition_raw_totals : list of (net_raw, proc_raw, compute_raw)
    actual_runtimes      : list of float  (same length, in matching order)

    Returns best params and full result grid sorted by error ascending.
    """
    results = []
    combos = list(product(c_net_vals, c_proc_vals, c_node_vals))

    for c_net, c_proc, c_node in tqdm(combos, desc="Grid search", unit="combo"):
        sq_err = 0.0
        for (nr, pr, cr), actual in zip(partition_raw_totals, actual_runtimes):
            est = c_net * nr + c_proc * pr + c_node * cr
            if actual > 0:
                sq_err += ((est - actual) / actual) ** 2
            else:
                sq_err += est ** 2
        mse = sq_err / len(actual_runtimes)
        results.append((mse, c_net, c_proc, c_node))

    results.sort()
    return results


def print_grid_results(results, partition_raw_totals, actual_runtimes, top_n=10):
    print(f"\n  Top {top_n} grid-search results (lowest relative MSE):")
    print(f"  {'Rank':>4}  {'c_net':>10} {'c_proc':>10} {'c_node':>10}  {'RMSE%':>8}  Predictions vs Actuals")
    print(f"  {'-'*4}  {'-'*10} {'-'*10} {'-'*10}  {'-'*8}  {'-'*40}")

    for rank, (mse, c_net, c_proc, c_node) in enumerate(results[:top_n], 1):
        preds = [
            c_net * nr + c_proc * pr + c_node * cr
            for nr, pr, cr in partition_raw_totals
        ]
        pred_str = "  ".join(f"{p:,.1f}" for p in preds)
        actual_str = "  ".join(f"{a:,.1f}" for a in actual_runtimes)
        rmse_pct = 100.0 * math.sqrt(mse)
        print(f"  {rank:>4}  {c_net:>10.2e} {c_proc:>10.2e} {c_node:>10.2e}  {rmse_pct:>7.2f}%  est=[{pred_str}]")

    print(f"         actual=[{actual_str}]")


# =============================================================================
# 6. Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Verify pregglenator cost estimates against trace data"
    )
    ap.add_argument("--trace", default="../comm_traces/src_0/merged.csv",
                    help="Single merged.csv trace file to load")
    ap.add_argument("--partitions", nargs="+",
                    default=["the_pregglenator_62000_v15/random.json"],
                    help="One or more partition JSON files to analyse")
    ap.add_argument("--c_net",  type=float, default=100000.0)
    ap.add_argument("--c_proc", type=float, default=10.0)
    ap.add_argument("--c_node", type=float, default=1.0)

    # Grid search
    ap.add_argument("--grid_search", action="store_true",
                    help="Grid search over cost parameters")
    ap.add_argument("--actual_runtimes", nargs="+", type=float, default=[-1.0, 1.0],
                    help="Observed runtimes (one per --partitions entry, same order). "
                         "Required for meaningful grid search error.")
    ap.add_argument("--grid_net",  nargs="+", type=float,
                    default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
                    help="c_net values to try")
    ap.add_argument("--grid_proc", nargs="+", type=float,
                    default=[0.1, 1.0, 10.0, 100.0, 1000.0],
                    help="c_proc values to try")
    ap.add_argument("--grid_node", nargs="+", type=float,
                    default=[0.01, 0.1, 1.0, 10.0, 100.0],
                    help="c_node values to try")
    ap.add_argument("--grid_top", type=int, default=10,
                    help="Number of top grid results to display")

    args = ap.parse_args()

    if args.actual_runtimes and len(args.actual_runtimes) != len(args.partitions):
        ap.error("--actual_runtimes must have the same number of entries as --partitions")

    # ── Load trace ────────────────────────────────────────────────────────────
    print("Loading trace...")
    out_presence, raw_recv_by_vertex, step_keys = load_trace(args.trace)
    print(f"  {len(step_keys)} task-steps  |  {len(out_presence):,} active source vertices")

    # ── Process each partition ────────────────────────────────────────────────
    all_labels      = []
    all_raw_totals  = []  # (net_raw_total, proc_raw_total, compute_raw_total)
    all_cost_totals = []  # (net_cost, proc_cost, compute_cost)

    for partition_path in args.partitions:
        print(f"\nLoading partition: {partition_path}")
        machine_of, worker_of = load_partition(partition_path)
        num_machines = (max(machine_of.values()) + 1) if machine_of else 1
        print(f"  {num_machines} machines  |  {len(set(worker_of.values()))} distinct worker IDs")

        net_raw, proc_raw, compute_raw, machine_loads = compute_raw_components(
            out_presence, raw_recv_by_vertex, machine_of, worker_of, step_keys
        )

        print_step_table(
            step_keys, net_raw, proc_raw, compute_raw, machine_loads,
            args.c_net, args.c_proc, args.c_node,
            partition_path, num_machines,
        )

        tn_raw = sum(net_raw.get(sk, 0.0)     for sk in step_keys)
        tp_raw = sum(proc_raw.get(sk, 0.0)    for sk in step_keys)
        tc_raw = sum(compute_raw.get(sk, 0.0) for sk in step_keys)

        all_labels.append(partition_path)
        all_raw_totals.append((tn_raw, tp_raw, tc_raw))
        all_cost_totals.append((
            args.c_net  * tn_raw,
            args.c_proc * tp_raw,
            args.c_node * tc_raw,
        ))

    # ── Comparison table (multiple partitions) ────────────────────────────────
    if len(args.partitions) > 1:
        print_comparison_table(all_labels, all_cost_totals)

    # ── Actual runtime comparison ─────────────────────────────────────────────
    if args.actual_runtimes:
        print(f"\n  Estimated vs Actual:")
        print(f"  {'Partition':<40}  {'Estimated':>14} {'Actual':>12} {'Ratio est/act':>14}")
        print(f"  {'-'*40}  {'-'*14} {'-'*12} {'-'*14}")
        for label, (tn, tp, tc), actual in zip(all_labels, all_cost_totals, args.actual_runtimes):
            est = tn + tp + tc
            ratio = est / actual if actual else float("inf")
            print(f"  {os.path.basename(label):<40}  {est:>14,.1f} {actual:>12,.3f} {ratio:>14.4f}")

    # ── Grid search ───────────────────────────────────────────────────────────
    if args.grid_search:
        n_combos = len(args.grid_net) * len(args.grid_proc) * len(args.grid_node)
        print(f"\nGrid search: {len(args.grid_net)} × {len(args.grid_proc)} × {len(args.grid_node)}"
              f" = {n_combos:,} combinations across {len(args.partitions)} partition(s)")

        actuals = args.actual_runtimes or [1.0] * len(args.partitions)
        if not args.actual_runtimes:
            print("  (no --actual_runtimes provided; showing relative cost ranking only)")

        results = grid_search(
            all_raw_totals, actuals,
            args.grid_net, args.grid_proc, args.grid_node,
        )

        print_grid_results(results, all_raw_totals, actuals, top_n=args.grid_top)

        best_mse, best_net, best_proc, best_node = results[0]
        print(f"\n  Best params:  c_net={best_net:.2e}  c_proc={best_proc:.2e}  c_node={best_node:.2e}")
        print(f"  RMSE:         {100.0 * math.sqrt(best_mse):.2f}%")
        print()
        print(f"  To re-run pregglenator with these params:")
        print(f"    python pregglenator_maximum_boogaloo.py "
              f"--c_net {best_net:.2e} --c_proc {best_proc:.2e} --c_node {best_node:.2e}")


if __name__ == "__main__":
    main()
