"""
visualizer and helper tool for pringle.
builds a synthetic graph, runs pringle on it and visualizes the output.

it compares 3 partitioning strategies:
  1) random baseline
  2) metis (topology-only): same edges, ignores comm volume (all existing edges weight=1)
  3) ours (weighted): the normal pregglenator run

it prints comm totals (evaluated on the ORIGINAL directed comm):
  - between-machine communication
  - within-machine between-worker communication

and shows TWO 2x3 figures:
  A) random vs ours
  B) metis(topology) vs ours
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
# load preggle output from json. yah its kinda gross
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
        ax.text(x0 + 0.15, y0 + H - 0.25, f"m{m}", fontsize=10)


def draw_worker_boxes(ax, worker_rects_by_m):
    for m, rects in worker_rects_by_m.items():
        for w, (x0, y0, W, H) in rects.items():
            rect = patches.Rectangle((x0, y0), W, H, fill=False, linewidth=1)
            ax.add_patch(rect)
            ax.text(x0 + 0.05, y0 + H - 0.18, f"w{w}", fontsize=8)


def set_bounds(ax, pos, pad=0.8):
    xs = [pos[u][0] for u in pos.keys()]
    ys = [pos[u][1] for u in pos.keys()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.axis("off")


# ============================================================
# 2x3 figure: top strategy vs bottom strategy
# ============================================================
def visualize_compare(
    und_adj,
    edges,
    top_machine,
    top_worker,
    bot_machine,
    bot_worker,
    num_machines,
    seed=0,
    top_label="top",
    bot_label="bottom",
):
    n = len(und_adj)

    base_pos = initial_layout(und_adj, edges_for_layout=edges, seed=seed)
    m_rects = machine_rects(num_machines)
    max_w = max([w for _, _, w in edges], default=1.0)

    # ----- top: machines then workers
    top_groups_m = groups_by_machine(n, top_machine)
    top_pos_m = remap_into_boxes(base_pos, top_groups_m, m_rects, seed=seed + 1)

    top_worker_ids_by_m = defaultdict(set)
    for u in range(n):
        top_worker_ids_by_m[top_machine[u]].add(top_worker[u])
    top_w_rects_by_m = {m: worker_rects_for_machine(top_worker_ids_by_m[m], m_rects[m]) for m in m_rects.keys()}

    top_groups_w = groups_by_worker_within_machine(n, top_machine, top_worker)
    top_rects_mw = {}
    for (m, w) in top_groups_w.keys():
        top_rects_mw[(m, w)] = top_w_rects_by_m[m][w]
    top_pos_w = remap_into_boxes(top_pos_m, top_groups_w, top_rects_mw, seed=seed + 2)

    # ----- bottom: machines then workers
    bot_groups_m = groups_by_machine(n, bot_machine)
    bot_pos_m = remap_into_boxes(base_pos, bot_groups_m, m_rects, seed=seed + 3)

    bot_worker_ids_by_m = defaultdict(set)
    for u in range(n):
        bot_worker_ids_by_m[bot_machine[u]].add(bot_worker[u])
    bot_w_rects_by_m = {m: worker_rects_for_machine(bot_worker_ids_by_m[m], m_rects[m]) for m in m_rects.keys()}

    bot_groups_w = groups_by_worker_within_machine(n, bot_machine, bot_worker)
    bot_rects_mw = {}
    for (m, w) in bot_groups_w.keys():
        bot_rects_mw[(m, w)] = bot_w_rects_by_m[m][w]
    bot_pos_w = remap_into_boxes(bot_pos_m, bot_groups_w, bot_rects_mw, seed=seed + 4)

    # plot
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    titles = ["initial graph", "machine boxes", "worker boxes"]

    # Top row
    axs[0][0].set_title(f"{top_label}: {titles[0]}")
    draw_edges(axs[0][0], edges, base_pos, max_w)
    draw_nodes(axs[0][0], base_pos, top_machine, node_size=18)
    set_bounds(axs[0][0], base_pos)

    axs[0][1].set_title(f"{top_label}: {titles[1]}")
    draw_machine_boxes(axs[0][1], m_rects)
    draw_edges(axs[0][1], edges, top_pos_m, max_w)
    draw_nodes(axs[0][1], top_pos_m, top_machine, node_size=18)
    set_bounds(axs[0][1], top_pos_m)

    axs[0][2].set_title(f"{top_label}: {titles[2]}")
    draw_machine_boxes(axs[0][2], m_rects)
    draw_worker_boxes(axs[0][2], top_w_rects_by_m)
    draw_edges(axs[0][2], edges, top_pos_w, max_w)
    draw_nodes(axs[0][2], top_pos_w, top_machine, node_size=18)
    set_bounds(axs[0][2], top_pos_w)

    # Bottom row
    axs[1][0].set_title(f"{bot_label}: {titles[0]}")
    draw_edges(axs[1][0], edges, base_pos, max_w)
    draw_nodes(axs[1][0], base_pos, bot_machine, node_size=18)
    set_bounds(axs[1][0], base_pos)

    axs[1][1].set_title(f"{bot_label}: {titles[1]}")
    draw_machine_boxes(axs[1][1], m_rects)
    draw_edges(axs[1][1], edges, bot_pos_m, max_w)
    draw_nodes(axs[1][1], bot_pos_m, bot_machine, node_size=18)
    set_bounds(axs[1][1], bot_pos_m)

    axs[1][2].set_title(f"{bot_label}: {titles[2]}")
    draw_machine_boxes(axs[1][2], m_rects)
    draw_worker_boxes(axs[1][2], bot_w_rects_by_m)
    draw_edges(axs[1][2], edges, bot_pos_w, max_w)
    draw_nodes(axs[1][2], bot_pos_w, bot_machine, node_size=18)
    set_bounds(axs[1][2], bot_pos_w)

    plt.tight_layout()
    plt.show()


# ============================================================
# Run pregglenator
# ============================================================
def run_pregglenator(pregglenator_path, comm_json_path, out_path, args):
    pregg = os.path.abspath(pregglenator_path)
    commp = os.path.abspath(comm_json_path)
    outp = os.path.abspath(out_path)

    cmd = [
        sys.executable,
        pregg,
        "--input",
        commp,
        "--format",
        "json",
        "--num_nodes",
        str(args.n),
        "--num_machines",
        str(args.num_machines),
        "--nodes_per_machine",
        str(args.nodes_per_machine),
        "--nodes_per_worker",
        str(args.nodes_per_worker),
        "--seed",
        str(args.seed),
        "--output",
        outp,
    ]

    print("\n=== running pregglenator ===")
    print("cwd:", os.getcwd())
    print("python:", sys.executable)
    print("cmd:", " ".join(cmd))
    print("out should be:", outp)

    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.stdout:
        print("\n=== pregglenator stdout ===")
        print(res.stdout)

    if res.stderr:
        print("\n=== pregglenator stderr ===")
        print(res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"pregglenator failed with return code {res.returncode}")

    if not os.path.exists(outp):
        raise RuntimeError(f"pregglenator returned success but did not write: {outp}")

    return outp


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregglenator", default="pregglenator.py")

    # Defaults
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--num_machines", type=int, default=4)
    ap.add_argument("--nodes_per_machine", type=int, default=20)
    ap.add_argument("--nodes_per_worker", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)

    # Trace generation knobs
    ap.add_argument("--p_edge", type=float, default=0.06)
    ap.add_argument("--intra_machine_bias", type=float, default=10.0)
    ap.add_argument("--intra_worker_bias", type=float, default=3.0)
    ap.add_argument("--weight_scale", type=int, default=50)

    # Viz knobs
    ap.add_argument("--top_k_edges", type=int, default=1500)
    ap.add_argument("--min_edge_weight", type=float, default=1.0)

    # Temp files
    ap.add_argument("--tmp_comm", default="_tmp_comm.json")
    ap.add_argument("--tmp_out", default="_tmp_assignment.json")
    ap.add_argument("--no_cleanup", action="store_true")

    args = ap.parse_args()

    if args.num_machines * args.nodes_per_machine < args.n:
        raise RuntimeError("Infeasible: NUM_MACHINES * NODES_PER_MACHINE < N")

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

    # 2) run ours (weighted) via pregglenator
    ours_out_path = run_pregglenator(args.pregglenator, args.tmp_comm, args.tmp_out, args)

    # 3) build undirected + edges for visualization
    comm_loaded = load_comm_json(args.tmp_comm)
    n = infer_num_nodes(comm_loaded, explicit_n=args.n)
    und_adj = symmetrize_to_undirected(comm_loaded, n)
    edges = top_edges_undirected(und_adj, top_k=args.top_k_edges, min_w=args.min_edge_weight)

    # 4) random baseline assignment
    rnd_machine, rnd_worker = random_capacity_partition(
        n=n,
        num_machines=args.num_machines,
        nodes_per_machine=args.nodes_per_machine,
        nodes_per_worker=args.nodes_per_worker,
        seed=args.seed,
    )

    # 5) ours assignment from pregglenator output
    ours_machine_map, ours_worker_map, ours_stats = load_assignment_json(ours_out_path)
    ours_machine = [ours_machine_map[u] for u in range(n)]
    ours_worker = [ours_worker_map[u] for u in range(n)]

    # 6) metis(topology-only): write a comm file with all weights=1 and run pregglenator again
    topo_comm = make_topology_only_comm_from_undirected(und_adj)
    topo_comm_path = "_tmp_comm_topology.json"
    topo_out_path = "_tmp_assignment_topology.json"
    dump_comm_json(topo_comm, topo_comm_path)
    topo_assignment_path = run_pregglenator(args.pregglenator, topo_comm_path, topo_out_path, args)

    topo_machine_map, topo_worker_map, topo_stats = load_assignment_json(topo_assignment_path)
    topo_machine = [topo_machine_map[u] for u in range(n)]
    topo_worker = [topo_worker_map[u] for u in range(n)]

    # 7) comparisons (evaluate on ORIGINAL directed comm)
    rnd_bm, rnd_bw, _ = comm_breakdown_directed(comm_loaded, rnd_machine, rnd_worker)
    topo_bm, topo_bw, _ = comm_breakdown_directed(comm_loaded, topo_machine, topo_worker)
    ours_bm, ours_bw, _ = comm_breakdown_directed(comm_loaded, ours_machine, ours_worker)

    print("\n=== COMPARISON (directed comm totals, evaluated on ORIGINAL comm) ===")
    print("Between-machine communication:")
    print("  random:         ", int(rnd_bm))
    print("  metis(topology):", int(topo_bm))
    print("  ours(weighted): ", int(ours_bm))

    print("Within-machine BETWEEN-worker communication:")
    print("  random:         ", int(rnd_bw))
    print("  metis(topology):", int(topo_bw))
    print("  ours(weighted): ", int(ours_bw))

    # 8) two figures
    visualize_compare(
        und_adj=und_adj,
        edges=edges,
        top_machine=rnd_machine,
        top_worker=rnd_worker,
        bot_machine=ours_machine,
        bot_worker=ours_worker,
        num_machines=args.num_machines,
        seed=args.seed,
        top_label="random baseline",
        bot_label="ours (weighted)",
    )

    visualize_compare(
        und_adj=und_adj,
        edges=edges,
        top_machine=topo_machine,
        top_worker=topo_worker,
        bot_machine=ours_machine,
        bot_worker=ours_worker,
        num_machines=args.num_machines,
        seed=args.seed,
        top_label="metis (topology-only)",
        bot_label="ours (weighted)",
    )

    # cleanup
    if not args.no_cleanup:
        for p in [args.tmp_comm, args.tmp_out, topo_comm_path, topo_out_path]:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()