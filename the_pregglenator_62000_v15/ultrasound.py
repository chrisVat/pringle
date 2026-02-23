"""
visualizer and helper tool for pringle. 
builds a synthetic graph, runs pringle on it and visualizes the output vs random partition
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



# build a fake graph 
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



# load preggle output from json. yah its kinda gross
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



# graph helpers
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



# random baseline 
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

        # chunk into workers
        num_workers = max(1, int(math.ceil(len(bucket) / float(nodes_per_worker))))
        for i, u in enumerate(bucket):
            w = i // nodes_per_worker
            worker_of[u] = min(w, num_workers - 1)

        # strict check
        counts = defaultdict(int)
        for u in bucket:
            counts[worker_of[u]] += 1
        if any(c > nodes_per_worker for c in counts.values()):
            raise RuntimeError("Random baseline violated worker cap (should not happen).")

    return machine_of, worker_of



# vis
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


def draw_edges(ax, edges, pos, max_w, alpha_min=0.01, alpha_max=0.8):
    """
    Make weak edges light and strong edges dark:
      - alpha scales with log(weight)
      - linewidth also scales with weight (mildly)
    """
    # log scale so weights with heavy tails don't dominate
    denom = math.log(1.0 + float(max_w)) if max_w > 0 else 1.0

    for u, v, w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        wn = math.log(1.0 + float(w)) / denom  # 0..1 roughly
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


def visualize_compare(
    und_adj,
    edges,
    rnd_machine,
    rnd_worker,
    met_machine,
    met_worker,
    num_machines,
    seed=0,
):
    n = len(und_adj)

    base_pos = initial_layout(und_adj, edges_for_layout=edges, seed=seed)

    m_rects = machine_rects(num_machines)

    # random: machines then workers
    rnd_groups_m = groups_by_machine(n, rnd_machine)
    rnd_pos_m = remap_into_boxes(base_pos, rnd_groups_m, m_rects, seed=seed + 1)

    rnd_worker_ids_by_m = defaultdict(set)
    for u in range(n):
        rnd_worker_ids_by_m[rnd_machine[u]].add(rnd_worker[u])
    rnd_w_rects_by_m = {m: worker_rects_for_machine(rnd_worker_ids_by_m[m], m_rects[m]) for m in m_rects.keys()}

    rnd_groups_w = groups_by_worker_within_machine(n, rnd_machine, rnd_worker)
    rnd_rects_mw = {}
    for (m, w) in rnd_groups_w.keys():
        rnd_rects_mw[(m, w)] = rnd_w_rects_by_m[m][w]
    rnd_pos_w = remap_into_boxes(rnd_pos_m, rnd_groups_w, rnd_rects_mw, seed=seed + 2)

    # ours: machines then workers
    met_groups_m = groups_by_machine(n, met_machine)
    met_pos_m = remap_into_boxes(base_pos, met_groups_m, m_rects, seed=seed + 3)

    met_worker_ids_by_m = defaultdict(set)
    for u in range(n):
        met_worker_ids_by_m[met_machine[u]].add(met_worker[u])
    met_w_rects_by_m = {m: worker_rects_for_machine(met_worker_ids_by_m[m], m_rects[m]) for m in m_rects.keys()}

    met_groups_w = groups_by_worker_within_machine(n, met_machine, met_worker)
    met_rects_mw = {}
    for (m, w) in met_groups_w.keys():
        met_rects_mw[(m, w)] = met_w_rects_by_m[m][w]
    met_pos_w = remap_into_boxes(met_pos_m, met_groups_w, met_rects_mw, seed=seed + 4)

    max_w = max([w for _, _, w in edges], default=1.0)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    titles = ["initial graph", "machine boxes", "worker boxes"]
    row_names = ["random baseline", "comm-based (pregglenator)"]

    # Row 0: random baseline
    axs[0][0].set_title(f"{row_names[0]}: {titles[0]}")
    draw_edges(axs[0][0], edges, base_pos, max_w)
    draw_nodes(axs[0][0], base_pos, rnd_machine, node_size=18)
    set_bounds(axs[0][0], base_pos)

    axs[0][1].set_title(f"{row_names[0]}: {titles[1]}")
    draw_machine_boxes(axs[0][1], m_rects)
    draw_edges(axs[0][1], edges, rnd_pos_m, max_w)
    draw_nodes(axs[0][1], rnd_pos_m, rnd_machine, node_size=18)
    set_bounds(axs[0][1], rnd_pos_m)

    axs[0][2].set_title(f"{row_names[0]}: {titles[2]}")
    draw_machine_boxes(axs[0][2], m_rects)
    draw_worker_boxes(axs[0][2], rnd_w_rects_by_m)
    draw_edges(axs[0][2], edges, rnd_pos_w, max_w)
    draw_nodes(axs[0][2], rnd_pos_w, rnd_machine, node_size=18)
    set_bounds(axs[0][2], rnd_pos_w)

    # Row 1: comm-based
    axs[1][0].set_title(f"{row_names[1]}: {titles[0]}")
    draw_edges(axs[1][0], edges, base_pos, max_w)
    draw_nodes(axs[1][0], base_pos, met_machine, node_size=18)
    set_bounds(axs[1][0], base_pos)

    axs[1][1].set_title(f"{row_names[1]}: {titles[1]}")
    draw_machine_boxes(axs[1][1], m_rects)
    draw_edges(axs[1][1], edges, met_pos_m, max_w)
    draw_nodes(axs[1][1], met_pos_m, met_machine, node_size=18)
    set_bounds(axs[1][1], met_pos_m)

    axs[1][2].set_title(f"{row_names[1]}: {titles[2]}")
    draw_machine_boxes(axs[1][2], m_rects)
    draw_worker_boxes(axs[1][2], met_w_rects_by_m)
    draw_edges(axs[1][2], edges, met_pos_w, max_w)
    draw_nodes(axs[1][2], met_pos_w, met_machine, node_size=18)
    set_bounds(axs[1][2], met_pos_w)

    plt.tight_layout()
    plt.show()


# ============================================================
# 10) Run pregglenator
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



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregglenator", default="pregglenator.py")

    # Defaults you asked for
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

    # 2) run pregglenator
    out_path = run_pregglenator(args.pregglenator, args.tmp_comm, args.tmp_out, args)

    # 3) load assignment + build random baseline
    comm_loaded = load_comm_json(args.tmp_comm)
    n = infer_num_nodes(comm_loaded, explicit_n=args.n)
    und_adj = symmetrize_to_undirected(comm_loaded, n)
    edges = top_edges_undirected(und_adj, top_k=args.top_k_edges, min_w=args.min_edge_weight)

    rnd_machine, rnd_worker = random_capacity_partition(
        n=n,
        num_machines=args.num_machines,
        nodes_per_machine=args.nodes_per_machine,
        nodes_per_worker=args.nodes_per_worker,
        seed=args.seed,
    )

    met_machine_map, met_worker_map, stats = load_assignment_json(out_path)
    met_machine = [met_machine_map[u] for u in range(n)]
    met_worker = [met_worker_map[u] for u in range(n)]

    # 4) comparisons (the two you requested)
    rnd_bm, rnd_bw, _ = comm_breakdown_directed(comm_loaded, rnd_machine, rnd_worker)
    met_bm, met_bw, _ = comm_breakdown_directed(comm_loaded, met_machine, met_worker)

    print("\n=== COMPARISON (directed comm totals) ===")
    print("Between-machine communication:")
    print("  random:", int(rnd_bm))
    print("  ours:  ", int(met_bm))
    print("Within-machine BETWEEN-worker communication:")
    print("  random:", int(rnd_bw))
    print("  ours:  ", int(met_bw))

    # 5) visualize (2x3)
    visualize_compare(
        und_adj=und_adj,
        edges=edges,
        rnd_machine=rnd_machine,
        rnd_worker=rnd_worker,
        met_machine=met_machine,
        met_worker=met_worker,
        num_machines=args.num_machines,
        seed=args.seed,
    )

    if not args.no_cleanup:
        for p in [args.tmp_comm, args.tmp_out]:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()