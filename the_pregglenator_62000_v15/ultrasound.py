"""
visualizer and helper tool for pringle.
builds a synthetic graph, runs partitioners on it and visualizes the output.

partitioning strategies:
  1) random baseline
  2) metis (topology-only): same edges, ignores comm volume (all existing edges weight=1)
  3) ours (weighted): the normal pregglenator run (two-level hierarchical)
  4) one-shot (two-level-aware): pregglenator_oneshot.py
  5) hypergraph (broadcast-aware): pregglenator_gamer_mode.py  (the v19 script you asked for)

it prints comm totals (evaluated on the ORIGINAL directed comm):
  - between-machine communication
  - within-machine between-worker communication
  - within-worker communication

and prints a simple total communication time estimate:
  - network: 400 us per "message unit"
  - process: 15 us per "message unit"

and shows ONE 1x3 figure per method (instead of 2x3 comparisons):
  columns: [initial graph] [machine boxes] [worker boxes]
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# build a fake graph: directed comm[src][dst] = weight
# ============================================================
def make_fake_comm_trace(
    n,
    num_machines,
    nodes_per_machine,
    nodes_per_worker,
    p_edge=0.06,
    intra_machine_bias=10.0,
    intra_worker_bias=3.0,
    weight_scale=50,
    seed=7,
):
    rng = random.Random(seed)

    true_machine = [min(i // nodes_per_machine, num_machines - 1) for i in range(n)]
    true_worker = []
    for i in range(n):
        m = true_machine[i]
        base = m * nodes_per_machine
        w = (i - base) // nodes_per_worker if nodes_per_worker > 0 else 0
        true_worker.append(w)

    comm = defaultdict(dict)

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if rng.random() > p_edge:
                continue

            x = rng.random()
            w = 1 + int(weight_scale * (x ** 2))

            if true_machine[u] == true_machine[v]:
                w = int(w * intra_machine_bias)
                if true_worker[u] == true_worker[v]:
                    w = int(w * intra_worker_bias)

            skew = 0.5 + rng.random()
            w = max(1, int(w * skew))

            comm[u][v] = comm[u].get(v, 0) + w

    return {int(s): {int(t): float(w) for t, w in nbrs.items()} for s, nbrs in comm.items()}


def dump_comm_json(comm, path):
    out = {}
    for s, nbrs in comm.items():
        out[str(int(s))] = {str(int(t)): float(w) for t, w in nbrs.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)


def load_comm_json(path):
    with open(path, "r") as f:
        raw = json.load(f)

    comm = {}
    for s, nbrs in raw.items():
        s_i = int(s)
        comm[s_i] = {}
        for t, w in nbrs.items():
            comm[s_i][int(t)] = float(w)
    return comm


# ============================================================
# load preggle output from json.
# ============================================================
def load_assignment_json(path):
    with open(path, "r") as f:
        payload = json.load(f)

    assign = payload["assignment"]
    machine_of = {}
    worker_of = {}
    for u_str, mw in assign.items():
        u = int(u_str)
        machine_of[u] = int(mw["machine"])
        worker_of[u] = int(mw["worker"])
    stats = payload.get("stats", {})
    return machine_of, worker_of, stats


# ============================================================
# graph helpers
# ============================================================
def infer_num_nodes(comm, explicit_n=None):
    if explicit_n is not None:
        return explicit_n
    mx = -1
    for s, nbrs in comm.items():
        mx = max(mx, int(s))
        for t in nbrs.keys():
            mx = max(mx, int(t))
    return mx + 1


def symmetrize_to_undirected(comm, n):
    """
    und[u][v] = comm[u][v] + comm[v][u]
    """
    adj = [defaultdict(float) for _ in range(n)]
    for u, nbrs in comm.items():
        if u < 0 or u >= n:
            continue
        for v, w in nbrs.items():
            if v < 0 or v >= n or v == u:
                continue
            adj[u][v] += float(w)

    und = [defaultdict(float) for _ in range(n)]
    for u in range(n):
        for v, w_uv in adj[u].items():
            w = w_uv + adj[v].get(u, 0.0)
            if w <= 0:
                continue
            und[u][v] = w
            und[v][u] = w
    return und


def top_edges_undirected(und_adj, top_k=1500, min_w=1.0):
    edges = []
    for u in range(len(und_adj)):
        for v, w in und_adj[u].items():
            if v <= u:
                continue
            if w < min_w:
                continue
            edges.append((u, v, float(w)))
    edges.sort(key=lambda x: x[2], reverse=True)
    if top_k is not None and len(edges) > top_k:
        edges = edges[:top_k]
    return edges


def make_topology_only_comm_from_undirected(und_adj):
    """
    Build a sparse directed comm dict that encodes only topology:
    for every undirected edge (u,v), emit a single directed edge u->v with weight=1.
    Then pregglenator will symmetrize anyway.
    """
    topo = defaultdict(dict)
    n = len(und_adj)
    for u in range(n):
        for v in und_adj[u].keys():
            if v <= u:
                continue
            topo[u][v] = 1.0
    return {int(s): {int(t): float(w) for t, w in nbrs.items()} for s, nbrs in topo.items()}


# ============================================================
# random baseline
# ============================================================
def random_capacity_partition(n, num_machines, nodes_per_machine, nodes_per_worker, seed=0):
    rng = random.Random(seed)

    if num_machines * nodes_per_machine < n:
        raise RuntimeError("Random baseline infeasible: NUM_MACHINES * NODES_PER_MACHINE < N")

    nodes = list(range(n))
    rng.shuffle(nodes)

    machine_of = [-1] * n
    cursor = 0
    for m in range(num_machines):
        take = min(nodes_per_machine, n - cursor)
        for u in nodes[cursor : cursor + take]:
            machine_of[u] = m
        cursor += take
        if cursor >= n:
            break

    if any(x == -1 for x in machine_of):
        raise RuntimeError("Random baseline infeasible: not enough capacity to place all nodes.")

    worker_of = [-1] * n
    for m in range(num_machines):
        bucket = [u for u in nodes if machine_of[u] == m]
        if not bucket:
            continue

        num_workers = max(1, int(math.ceil(len(bucket) / float(nodes_per_worker))))
        for i, u in enumerate(bucket):
            w = i // nodes_per_worker
            worker_of[u] = min(w, num_workers - 1)

        counts = defaultdict(int)
        for u in bucket:
            counts[worker_of[u]] += 1
        if any(c > nodes_per_worker for c in counts.values()):
            raise RuntimeError("Random baseline violated worker cap (should not happen).")

    return machine_of, worker_of


# ============================================================
# metrics: evaluate on ORIGINAL directed comm
# ============================================================
def comm_breakdown_directed(comm, machine_of, worker_of):
    between_machines = 0.0
    within_machine_between_workers = 0.0
    within_worker = 0.0

    for s, nbrs in comm.items():
        ms = machine_of[s]
        ws = worker_of[s]
        for t, w in nbrs.items():
            mt = machine_of[t]
            wt = worker_of[t]
            if ms != mt:
                between_machines += w
            else:
                if ws != wt:
                    within_machine_between_workers += w
                else:
                    within_worker += w

    return between_machines, within_machine_between_workers, within_worker


def estimate_total_comm_time_seconds(between_machine, within_machine_between_worker, network_us=300.0, process_us=15.0):
    """
    Crude estimate:
      total_time = between_machine * network_us + within_machine_between_worker * process_us
    Treats the comm weights as "message units".
    """
    total_us = between_machine * network_us + within_machine_between_worker * process_us
    return total_us / 1e6


# ============================================================
# layout + remapping into boxes
# ============================================================
def initial_layout(und_adj, edges_for_layout, seed=0):
    try:
        import networkx as nx

        G = nx.Graph()
        for u, v, w in edges_for_layout:
            G.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G, seed=seed, weight="weight", iterations=200)

        n = len(und_adj)
        rng = random.Random(seed)
        for u in range(n):
            if u not in pos:
                pos[u] = (rng.random(), rng.random())

        return {u: (float(pos[u][0]), float(pos[u][1])) for u in range(n)}
    except Exception:
        n = len(und_adj)
        rng = random.Random(seed)
        pos = [(rng.random(), rng.random()) for _ in range(n)]

        iters = 300
        dt = 0.02
        k_att = 0.0008

        for _ in range(iters):
            fx = [0.0] * n
            fy = [0.0] * n

            for u, v, w in edges_for_layout:
                xu, yu = pos[u]
                xv, yv = pos[v]
                dx = xv - xu
                dy = yv - yu
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                f = k_att * math.log(1.0 + w) * dist
                fx[u] += dx * f
                fy[u] += dy * f
                fx[v] -= dx * f
                fy[v] -= dy * f

            pos = [(pos[i][0] + dt * fx[i], pos[i][1] + dt * fy[i]) for i in range(n)]

        return {u: (pos[u][0], pos[u][1]) for u in range(n)}


def normalize_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = max(1e-9, maxx - minx)
    dy = max(1e-9, maxy - miny)
    return [((p[0] - minx) / dx, (p[1] - miny) / dy) for p in points]


def machine_rects(num_machines, box_w=10.0, box_h=6.0, pad=1.5):
    cols = int(math.ceil(math.sqrt(num_machines)))
    rows = int(math.ceil(num_machines / float(cols)))

    rects = {}
    for m in range(num_machines):
        r = m // cols
        c = m % cols
        x0 = c * (box_w + pad)
        y0 = (rows - 1 - r) * (box_h + pad)
        rects[m] = (x0, y0, box_w, box_h)
    return rects


def worker_rects_for_machine(worker_ids, m_rect, pad=0.25):
    x0, y0, W, H = m_rect
    inner_x0 = x0 + pad
    inner_y0 = y0 + pad
    inner_W = W - 2 * pad
    inner_H = H - 2 * pad

    ws = sorted(worker_ids)
    k = len(ws)
    cols = max(1, int(math.ceil(math.sqrt(k))))
    rows = int(math.ceil(k / float(cols)))

    rects = {}
    cell_w = inner_W / float(cols)
    cell_h = inner_H / float(rows)

    for i, w in enumerate(ws):
        r = i // cols
        c = i % cols
        wx0 = inner_x0 + c * cell_w + pad
        wy0 = inner_y0 + (rows - 1 - r) * cell_h + pad
        rects[w] = (wx0, wy0, max(0.1, cell_w - 2 * pad), max(0.1, cell_h - 2 * pad))

    return rects


def remap_into_boxes(base_pos, groups, rects, jitter=0.02, seed=0):
    rng = random.Random(seed)
    new_pos = {}

    for key, nodes in groups.items():
        x0, y0, W, H = rects[key]
        pts = [base_pos[u] for u in nodes]

        if len(nodes) == 1:
            u = nodes[0]
            new_pos[u] = (x0 + 0.5 * W, y0 + 0.5 * H)
            continue

        norm = normalize_points(pts)
        for u, (nx, ny) in zip(nodes, norm):
            jx = (rng.random() - 0.5) * jitter
            jy = (rng.random() - 0.5) * jitter
            new_pos[u] = (x0 + (nx + jx) * W, y0 + (ny + jy) * H)

    return new_pos


def groups_by_machine(n, machine_of):
    g = defaultdict(list)
    for u in range(n):
        g[machine_of[u]].append(u)
    return dict(g)


def groups_by_worker_within_machine(n, machine_of, worker_of):
    g = defaultdict(list)  # key=(m,w)
    for u in range(n):
        g[(machine_of[u], worker_of[u])].append(u)
    return dict(g)


# ============================================================
# drawing
# ============================================================
def draw_edges(ax, edges, pos, max_w, alpha_min=0.01, alpha_max=0.8):
    """
    weak edges fade out, strong edges pop.
    alpha scales with log(weight), linewidth scales mildly too.
    """
    denom = math.log(1.0 + float(max_w)) if max_w > 0 else 1.0

    for u, v, w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        wn = math.log(1.0 + float(w)) / denom
        a = alpha_min + (alpha_max - alpha_min) * (wn ** 1.7)
        lw = 0.10 + 2.6 * (wn ** 1.2)

        ax.plot([x1, x2], [y1, y2], linewidth=lw, alpha=a)


def draw_nodes(ax, pos, machine_of, node_size=18):
    nodes = sorted(pos.keys())
    xs = [pos[u][0] for u in nodes]
    ys = [pos[u][1] for u in nodes]
    cs = [machine_of[u] for u in nodes]
    ax.scatter(xs, ys, c=cs, s=node_size, zorder=3)


def draw_machine_boxes(ax, m_rects):
    for m, (x0, y0, W, H) in m_rects.items():
        rect = patches.Rectangle((x0, y0), W, H, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0 + 0.15, y0 + H - 0.35, f"m{m}", fontsize=10)


def draw_worker_boxes(ax, worker_rects_by_m):
    for m, rects in worker_rects_by_m.items():
        for w, (x0, y0, W, H) in rects.items():
            rect = patches.Rectangle((x0, y0), W, H, fill=False, linewidth=1)
            ax.add_patch(rect)
            ax.text(x0 + 0.05, y0 + H - 0.28, f"w{w}", fontsize=8)


def set_bounds(ax, pos, pad=0.8):
    xs = [pos[u][0] for u in pos.keys()]
    ys = [pos[u][1] for u in pos.keys()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.axis("off")


# ============================================================
# 1x3 figure per method
# ============================================================
def visualize_method(
    und_adj,
    edges,
    machine,
    worker,
    num_machines,
    seed=0,
    label="method",
):
    n = len(und_adj)

    base_pos = initial_layout(und_adj, edges_for_layout=edges, seed=seed)
    m_rects = machine_rects(num_machines)
    max_w = max([w for _, _, w in edges], default=1.0)

    # machines then workers
    groups_m = groups_by_machine(n, machine)
    pos_m = remap_into_boxes(base_pos, groups_m, m_rects, seed=seed + 1)

    worker_ids_by_m = defaultdict(set)
    for u in range(n):
        worker_ids_by_m[machine[u]].add(worker[u])
    w_rects_by_m = {m: worker_rects_for_machine(worker_ids_by_m[m], m_rects[m]) for m in m_rects.keys()}

    groups_w = groups_by_worker_within_machine(n, machine, worker)
    rects_mw = {}
    for (m, w) in groups_w.keys():
        rects_mw[(m, w)] = w_rects_by_m[m][w]
    pos_w = remap_into_boxes(pos_m, groups_w, rects_mw, seed=seed + 2)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["initial graph", "machine boxes", "worker boxes"]

    axs[0].set_title(f"{label}: {titles[0]}")
    draw_edges(axs[0], edges, base_pos, max_w)
    draw_nodes(axs[0], base_pos, machine, node_size=18)
    set_bounds(axs[0], base_pos)

    axs[1].set_title(f"{label}: {titles[1]}")
    draw_machine_boxes(axs[1], m_rects)
    draw_edges(axs[1], edges, pos_m, max_w)
    draw_nodes(axs[1], pos_m, machine, node_size=18)
    set_bounds(axs[1], pos_m)

    axs[2].set_title(f"{label}: {titles[2]}")
    draw_machine_boxes(axs[2], m_rects)
    draw_worker_boxes(axs[2], w_rects_by_m)
    draw_edges(axs[2], edges, pos_w, max_w)
    draw_nodes(axs[2], pos_w, machine, node_size=18)
    set_bounds(axs[2], pos_w)

    plt.tight_layout()
    plt.show()


# ============================================================
# Run partitioners
# ============================================================
def _run_script(script_path, comm_json_path, out_path, cmd_args, banner):
    script_abs = os.path.abspath(script_path)
    comm_abs = os.path.abspath(comm_json_path)
    out_abs = os.path.abspath(out_path)

    cmd = [sys.executable, script_abs] + [
        "--input", comm_abs,
        "--format", "json",
    ] + cmd_args + [
        "--output", out_abs,
    ]

    print(f"\n=== running {banner} ===")
    print("cwd:", os.getcwd())
    print("python:", sys.executable)
    print("cmd:", " ".join(cmd))
    print("out should be:", out_abs)

    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.stdout:
        print(f"\n=== {banner} stdout ===")
        print(res.stdout)

    if res.stderr:
        print(f"\n=== {banner} stderr ===")
        print(res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"{banner} failed with return code {res.returncode}")

    if not os.path.exists(out_abs):
        raise RuntimeError(f"{banner} returned success but did not write: {out_abs}")

    return out_abs


def run_pregglenator_hier(pregglenator_path, comm_json_path, out_path, args):
    cmd_args = [
        "--num_nodes", str(args.n),
        "--num_machines", str(args.num_machines),
        "--nodes_per_machine", str(args.nodes_per_machine),
        "--nodes_per_worker", str(args.nodes_per_worker),
        "--seed", str(args.seed),
    ]
    return _run_script(pregglenator_path, comm_json_path, out_path, cmd_args, banner="pregglenator (hier)")


def run_pregglenator_oneshot(pregglenator_oneshot_path, comm_json_path, out_path, args):
    cmd_args = [
        "--num_nodes", str(args.n),
        "--num_machines", str(args.num_machines),
        "--nodes_per_machine", str(args.nodes_per_machine),
        "--nodes_per_worker", str(args.nodes_per_worker),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--max_refine_passes", str(args.max_refine_passes),
        "--seed", str(args.seed),
    ]
    return _run_script(pregglenator_oneshot_path, comm_json_path, out_path, cmd_args, banner="pregglenator (one-shot)")


def run_pregglenator_hypergraph(pregglenator_hypergraph_path, comm_json_path, out_path, args):
    cmd_args = [
        "--num_nodes", str(args.n),
        "--num_machines", str(args.num_machines),
        "--nodes_per_machine", str(args.nodes_per_machine),
        "--nodes_per_worker", str(args.nodes_per_worker),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--seed", str(args.seed),
        # keep these aligned with oneshot knobs for fair runtime, if present in v19
        "--alt_rounds", str(args.alt_rounds),
        "--remap_rounds", str(args.remap_rounds),
        "--sa_passes", str(args.sa_passes),
        "--sa_T_decay", str(args.sa_T_decay),
        "--slack_factor", str(args.slack_factor),
        "--lambda_worker", str(args.lambda_worker),
        "--lambda_machine", str(args.lambda_machine),
        "--coarsen_percentile", str(args.coarsen_percentile),
        "--coarsen_max_pair_fraction", str(args.coarsen_max_pair_fraction),
    ]
    if args.sa_steps_per_pass is not None:
        cmd_args += ["--sa_steps_per_pass", str(args.sa_steps_per_pass)]
    if args.sa_T0 is not None:
        cmd_args += ["--sa_T0", str(args.sa_T0)]
    if args.final_strict_repair:
        cmd_args += ["--final_strict_repair"]
    return _run_script(pregglenator_hypergraph_path, comm_json_path, out_path, cmd_args, banner="pregglenator (hypergraph)")


# ============================================================
# pretty printing
# ============================================================
def _fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def _fmt_time_s(x):
    return f"{x:.3f}s"


from collections import defaultdict
import statistics

def print_comparison(comm_loaded, n, assignments, ref_name="one_shot", network_us=400.0, process_us=15.0):
    """
    assignments: dict name -> (machine_list, worker_list)
    """
    rows = {}
    for name, (m_of, w_of) in assignments.items():
        # 1. Base Raw Breakdown
        bm, bw, ww = comm_breakdown_directed(comm_loaded, m_of, w_of)
        
        # 2. Sequential/Serial Time Estimate (Legacy)
        serial_t_s = (bm * network_us + bw * process_us) / 1e6
        
        # 3. Per-machine / Parallel Tracking
        machine_packets = defaultdict(set)
        machine_ipc_cost = defaultdict(float)
        machine_compute = defaultdict(float)
        machine_net_msgs = defaultdict(float)
        
        for u, nbrs in comm_loaded.items():
            u_i = int(u)
            if u_i >= len(m_of): continue
            mu = m_of[u_i]
            wu = w_of[u_i]
            
            for v, w in nbrs.items():
                v_i = int(v)
                if v_i >= len(m_of): continue
                mv = m_of[v_i]
                wv = w_of[v_i]
                wf = float(w)
                
                machine_compute[mu] += wf
                machine_compute[mv] += wf
                
                if mu != mv:
                    machine_packets[mu].add(mv)
                    machine_net_msgs[mu] += wf
                elif wu != wv:
                    machine_ipc_cost[mu] += wf
                    
        # Calculate individual machine times for Parallel/Straggler view
        m_times = {}
        all_ms = sorted(list(set(m_of)))
        for m in all_ms:
            pkts = len(machine_packets[m])
            ipc = machine_ipc_cost[m]
            m_times[m] = (pkts * network_us + ipc * process_us) / 1e6

        times_list = list(m_times.values())
        max_t = max(times_list) if times_list else 0
        avg_t = sum(times_list) / len(times_list) if times_list else 0
        imbalance = max_t / avg_t if avg_t > 0 else 1.0
        
        peak_m = -1
        max_comp = 0.0
        if machine_compute:
            peak_m = max(machine_compute, key=machine_compute.get)
            max_comp = machine_compute[peak_m]

        rows[name] = {
            "between_machine": bm,
            "between_worker": bw,
            "within_worker": ww,
            "serial_time_s": serial_t_s,
            "parallel_time_s": max_t,
            "m_times": m_times,
            "avg_t": avg_t,
            "imbalance": imbalance,
            "max_compute": max_comp,
            "peak_m_packets": len(machine_packets[peak_m]),
            "peak_m_net_msgs": machine_net_msgs[peak_m],
            "peak_m_ipc": machine_ipc_cost[peak_m]
        }

    # --- FINAL PRINTING BLOCK ---
    print("\n=== COMPARISON (directed comm totals, evaluated on ORIGINAL comm) ===")
    print("Between-machine communication (Raw Message Count):")
    for name in assignments.keys():
        print(f"  {name:<15} {_fmt_int(rows[name]['between_machine'])}")

    print("Within-machine BETWEEN-worker communication:")
    for name in assignments.keys():
        print(f"  {name:<15} {_fmt_int(rows[name]['between_worker'])}")

    print("Within-worker communication:")
    for name in assignments.keys():
        print(f"  {name:<15} {_fmt_int(rows[name]['within_worker'])}")

    print("\n=== TOTAL COMMUNICATION TIME ESTIMATE (theoretical serial time) ===")
    print(f"Assumptions: network={network_us:.1f}us per unit, process={process_us:.1f}us per unit")
    for name in assignments.keys():
        print(f"  {name:<15} {_fmt_time_s(rows[name]['serial_time_s'])}")

    print("\n=== PER-MACHINE COMMUNICATION TIME (Parallel Packet + IPC) ===")
    for name in assignments.keys():
        print(f"  {name.upper()}:")
        m_data = rows[name]['m_times']
        for m_id, t in m_data.items():
            print(f"   m{m_id}: {_fmt_time_s(t)}", end=", ")
        print(f"   {'Avg:':<8} {_fmt_time_s(rows[name]['avg_t'])}")
        print(f"   {'Imbal:':<8} {rows[name]['imbalance']:.3f}x")

    print("\n=== PARALLEL COMMUNICATION TIME (assumes perfect combiners and network parallelism ===")
    print(f"Assumptions: network={network_us:.1f}us per PACKET, process={process_us:.1f}us per MSG")
    for name in assignments.keys():
        print(f"  {name:<15} {_fmt_time_s(rows[name]['parallel_time_s'])}")

    print("\n=== Peak message load on a single machine ===")
    print("Format: Total Msgs (Raw Net Msgs, Combined Packets, IPC Msgs)")
    for name in assignments.keys():
        r = rows[name]
        print(f"  {name:<15} {_fmt_int(r['max_compute']):>8} (NetMsgs: {_fmt_int(r['peak_m_net_msgs'])}, Pkts: {int(r['peak_m_packets'])}, IPC: {_fmt_int(r['peak_m_ipc'])})")

    if ref_name in rows:
        ref = rows[ref_name]
        print(f"\n=== RELATIVE TO {ref_name.upper()} (Sync Barrier Time) ===")
        for name in assignments.keys():
            if name == ref_name: continue
            ratio = rows[name]["parallel_time_s"] / max(1e-12, ref["parallel_time_s"])
            print(f"  {name:<15} bottleneck_ratio={ratio:.3f}x (lower is better)")

    return rows


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregglenator", default="pregglenator.py")
    ap.add_argument("--pregglenator_oneshot", default="pregglenator_oneshot.py")
    ap.add_argument("--pregglenator_hypergraph", default="pregglenator_gamer_mode.py")

    # Defaults
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--num_machines", type=int, default=4)
    ap.add_argument("--nodes_per_machine", type=int, default=20)
    ap.add_argument("--nodes_per_worker", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)

    # One-shot knobs (passed through to oneshot/hypergraph where applicable)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max_refine_passes", type=int, default=18)

    # Hypergraph v19 knobs (optional but exposed for parity)
    ap.add_argument("--alt_rounds", type=int, default=10)
    ap.add_argument("--remap_rounds", type=int, default=12)
    ap.add_argument("--sa_passes", type=int, default=10)
    ap.add_argument("--sa_steps_per_pass", type=int, default=None)
    ap.add_argument("--sa_T0", type=float, default=None)
    ap.add_argument("--sa_T_decay", type=float, default=0.85)
    ap.add_argument("--slack_factor", type=float, default=1.05)
    ap.add_argument("--lambda_worker", type=float, default=10.0)
    ap.add_argument("--lambda_machine", type=float, default=10.0)
    ap.add_argument("--coarsen_percentile", type=float, default=95.0)
    ap.add_argument("--coarsen_max_pair_fraction", type=float, default=0.45)
    ap.add_argument("--final_strict_repair", action="store_true")

    # Trace generation knobs
    ap.add_argument("--p_edge", type=float, default=0.06)
    ap.add_argument("--intra_machine_bias", type=float, default=10.0)
    ap.add_argument("--intra_worker_bias", type=float, default=3.0)
    ap.add_argument("--weight_scale", type=int, default=50)

    # Viz knobs
    ap.add_argument("--top_k_edges", type=int, default=1500)
    ap.add_argument("--min_edge_weight", type=float, default=1.0)

    # Time estimate knobs
    ap.add_argument("--network_us", type=float, default=400.0)
    ap.add_argument("--process_us", type=float, default=15.0)

    # Temp files
    ap.add_argument("--tmp_comm", default="_tmp_comm.json")
    ap.add_argument("--tmp_out_ours", default="_tmp_assignment_ours.json")
    ap.add_argument("--tmp_out_topo", default="_tmp_assignment_topology.json")
    ap.add_argument("--tmp_out_oneshot", default="_tmp_assignment_oneshot.json")
    ap.add_argument("--tmp_out_hyper", default="_tmp_assignment_hypergraph.json")
    ap.add_argument("--tmp_comm_topo", default="_tmp_comm_topology.json")
    ap.add_argument("--no_cleanup", action="store_true")

    args = ap.parse_args()

    if args.num_machines * args.nodes_per_machine < args.n:
        raise RuntimeError("Infeasible: NUM_MACHINES * NODES_PER_MACHINE < N")

    if args.alpha < args.beta:
        raise RuntimeError("Require alpha >= beta (cross-machine penalty should be >= intra-machine).")

    # 1) generate trace
    comm = make_fake_comm_trace(
        n=args.n,
        num_machines=args.num_machines,
        nodes_per_machine=args.nodes_per_machine,
        nodes_per_worker=args.nodes_per_worker,
        p_edge=args.p_edge,
        intra_machine_bias=args.intra_machine_bias,
        intra_worker_bias=args.intra_worker_bias,
        weight_scale=args.weight_scale,
        seed=args.seed,
    )
    dump_comm_json(comm, args.tmp_comm)

    # 2) run ours (weighted, hierarchical)
    ours_out_path = run_pregglenator_hier(args.pregglenator, args.tmp_comm, args.tmp_out_ours, args)

    # 3) run one-shot
    oneshot_out_path = run_pregglenator_oneshot(args.pregglenator_oneshot, args.tmp_comm, args.tmp_out_oneshot, args)

    # 4) run hypergraph (broadcast-aware)
    hyper_out_path = run_pregglenator_hypergraph(args.pregglenator_hypergraph, args.tmp_comm, args.tmp_out_hyper, args)

    # 5) build undirected + edges for visualization
    comm_loaded = load_comm_json(args.tmp_comm)
    n = infer_num_nodes(comm_loaded, explicit_n=args.n)
    und_adj = symmetrize_to_undirected(comm_loaded, n)
    edges = top_edges_undirected(und_adj, top_k=args.top_k_edges, min_w=args.min_edge_weight)

    # 6) random baseline assignment
    rnd_machine, rnd_worker = random_capacity_partition(
        n=n,
        num_machines=args.num_machines,
        nodes_per_machine=args.nodes_per_machine,
        nodes_per_worker=args.nodes_per_worker,
        seed=args.seed,
    )

    # 7) ours assignment
    ours_machine_map, ours_worker_map, _ = load_assignment_json(ours_out_path)
    ours_machine = [ours_machine_map[u] for u in range(n)]
    ours_worker = [ours_worker_map[u] for u in range(n)]

    # 8) oneshot assignment
    oneshot_machine_map, oneshot_worker_map, _ = load_assignment_json(oneshot_out_path)
    oneshot_machine = [oneshot_machine_map[u] for u in range(n)]
    oneshot_worker = [oneshot_worker_map[u] for u in range(n)]

    # 9) hypergraph assignment
    hyper_machine_map, hyper_worker_map, hyper_stats = load_assignment_json(hyper_out_path)
    hyper_machine = [hyper_machine_map[u] for u in range(n)]
    hyper_worker = [hyper_worker_map[u] for u in range(n)]

    # 10) metis(topology-only): write comm with all weights=1 and run hierarchical again
    topo_comm = make_topology_only_comm_from_undirected(und_adj)
    dump_comm_json(topo_comm, args.tmp_comm_topo)
    topo_assignment_path = run_pregglenator_hier(args.pregglenator, args.tmp_comm_topo, args.tmp_out_topo, args)

    topo_machine_map, topo_worker_map, _ = load_assignment_json(topo_assignment_path)
    topo_machine = [topo_machine_map[u] for u in range(n)]
    topo_worker = [topo_worker_map[u] for u in range(n)]

    # 11) comparisons (evaluate on ORIGINAL directed comm)
    assignments = {
        "random": (rnd_machine, rnd_worker),
        "metis_topo": (topo_machine, topo_worker),
        "ours": (ours_machine, ours_worker),
        "one_shot": (oneshot_machine, oneshot_worker),
        "hypergraph": (hyper_machine, hyper_worker),
    }

    print_comparison(
        comm_loaded=comm_loaded,
        n=n,
        assignments=assignments,
        ref_name="one_shot",
        network_us=args.network_us,
        process_us=args.process_us,
    )

    # 12) show 1x3 per method
    visualize_method(
        und_adj=und_adj,
        edges=edges,
        machine=rnd_machine,
        worker=rnd_worker,
        num_machines=args.num_machines,
        seed=args.seed,
        label="random baseline",
    )

    visualize_method(
        und_adj=und_adj,
        edges=edges,
        machine=topo_machine,
        worker=topo_worker,
        num_machines=args.num_machines,
        seed=args.seed,
        label="metis (topology-only)",
    )

    visualize_method(
        und_adj=und_adj,
        edges=edges,
        machine=ours_machine,
        worker=ours_worker,
        num_machines=args.num_machines,
        seed=args.seed,
        label="ours (weighted, hier)",
    )

    visualize_method(
        und_adj=und_adj,
        edges=edges,
        machine=oneshot_machine,
        worker=oneshot_worker,
        num_machines=args.num_machines,
        seed=args.seed,
        label=f"one-shot (alpha={args.alpha}, beta={args.beta})",
    )

    visualize_method(
        und_adj=und_adj,
        edges=edges,
        machine=hyper_machine,
        worker=hyper_worker,
        num_machines=args.num_machines,
        seed=args.seed,
        label=f"hypergraph (alpha={args.alpha}, beta={args.beta})",
    )

    # cleanup
    if not args.no_cleanup:
        for p in [
            args.tmp_comm,
            args.tmp_comm_topo,
            args.tmp_out_ours,
            args.tmp_out_topo,
            args.tmp_out_oneshot,
            args.tmp_out_hyper,
        ]:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()