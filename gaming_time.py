#!/usr/bin/env python3
"""

See gamers.txt before running this

Representative SSSP source selection for SNAP twitch_gamers.

What it does (scalable, no NetworkX):
1) Read node ids from large_twitch_features.csv (so isolates exist).
2) Stream large_twitch_edges.csv once:
   - degree per node
   - union-find for connected components
   - store edges as int32 endpoint indices
3) Build CSR adjacency (undirected).
4) Select MAX_SELECTED nodes via:
   - hubs (high degree, component-capped)
   - isolates/fringe (deg 0/1/2, component-capped)
   - mid-degree (log-binned stratified, component-capped)
   - far-apart seeds inside giant component (few BFS passes, farthest-first)

Outputs:
  selected_nodes.txt  (original ids, one per line)
  selected_nodes.json (json list)

Usage:
  python select_sources.py \
    --features twitch_gamers/large_twitch_features.csv \
    --edges    twitch_gamers/large_twitch_edges.csv \
    --max_selected 64 \
    --seed 0 \
    --out_dir .

"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict, deque

import numpy as np


# ----------------------------
# DSU
# ----------------------------
class DSU:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ----------------------------
# CSV parsing helpers
# ----------------------------
def _looks_like_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def detect_id_column(header):
    if header is None:
        return 0
    lowered = [h.strip().lower() for h in header]
    for key in ["numeric_id", "node_id", "id", "userid", "user_id"]:
        if key in lowered:
            return lowered.index(key)
    return 0


def read_feature_node_ids(features_path: str):
    node_ids = []
    with open(features_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError("Empty features file")

        if _looks_like_int(first[0]):
            # no header
            node_ids.append(int(first[0]))
            for row in reader:
                if row:
                    node_ids.append(int(row[0]))
        else:
            # header
            header = first
            id_col = detect_id_column(header)
            for row in reader:
                if row:
                    node_ids.append(int(row[id_col]))

    node_ids = np.asarray(node_ids, dtype=np.int64)
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
    return node_ids, id_to_idx


def detect_edge_columns(header):
    """
    Returns indices (c0, c1) of the two endpoint columns.
    If known names exist, use them; else use first two columns.
    """
    lowered = [h.strip().lower() for h in header]
    candidates = [
        ("numeric_id_1", "numeric_id_2"),
        ("from", "to"),
        ("src", "dst"),
        ("source", "target"),
        ("u", "v"),
        ("node1", "node2"),
        ("id_1", "id_2"),
    ]
    for a, b in candidates:
        if a in lowered and b in lowered:
            return lowered.index(a), lowered.index(b)
    return 0, 1


def read_edges_degree_components(edges_path: str, id_to_idx: dict, n: int):
    dsu = DSU(n)
    deg = np.zeros(n, dtype=np.int32)

    us = []
    vs = []

    with open(edges_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError("Empty edges file")

        if _looks_like_int(first[0]) and _looks_like_int(first[1]):
            # no header
            edge_c0, edge_c1 = 0, 1

            def handle_row(row):
                a = int(row[edge_c0])
                b = int(row[edge_c1])
                ia = id_to_idx.get(a, None)
                ib = id_to_idx.get(b, None)
                if ia is None or ib is None:
                    return
                us.append(ia)
                vs.append(ib)
                deg[ia] += 1
                deg[ib] += 1
                dsu.union(ia, ib)

            handle_row(first)
            for row in reader:
                if row and len(row) >= 2:
                    handle_row(row)

        else:
            # header
            header = first
            edge_c0, edge_c1 = detect_edge_columns(header)

            def handle_row(row):
                a = int(row[edge_c0])
                b = int(row[edge_c1])
                ia = id_to_idx.get(a, None)
                ib = id_to_idx.get(b, None)
                if ia is None or ib is None:
                    return
                us.append(ia)
                vs.append(ib)
                deg[ia] += 1
                deg[ib] += 1
                dsu.union(ia, ib)

            for row in reader:
                if row and len(row) > max(edge_c0, edge_c1):
                    handle_row(row)

    u = np.asarray(us, dtype=np.int32)
    v = np.asarray(vs, dtype=np.int32)
    return u, v, deg, dsu


# ----------------------------
# CSR + BFS
# ----------------------------
def build_csr_undirected(n: int, u: np.ndarray, v: np.ndarray):
    counts = np.bincount(u, minlength=n).astype(np.int64) + np.bincount(v, minlength=n).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(indptr[-1], dtype=np.int32)

    cursor = indptr[:-1].copy()
    m = u.shape[0]
    for i in range(m):
        a = int(u[i])
        b = int(v[i])
        indices[cursor[a]] = b
        cursor[a] += 1
        indices[cursor[b]] = a
        cursor[b] += 1

    return indptr, indices


def bfs_farthest(indptr, indices, start: int, allowed_mask=None):
    """
    BFS from start. Returns:
      farthest_node, dist_array (int32, -1 unvisited), max_dist
    If allowed_mask is provided (bool array), BFS only traverses nodes where allowed_mask[node]=True.
    """
    n = allowed_mask.shape[0] if allowed_mask is not None else (indptr.shape[0] - 1)
    dist = np.full(n, -1, dtype=np.int32)

    if allowed_mask is not None and not allowed_mask[start]:
        return start, dist, 0

    q = deque([start])
    dist[start] = 0
    farthest = start
    maxd = 0

    while q:
        x = q.popleft()
        dx = dist[x]
        if dx > maxd:
            maxd = dx
            farthest = x
        for j in range(indptr[x], indptr[x + 1]):
            y = int(indices[j])
            if dist[y] != -1:
                continue
            if allowed_mask is not None and not allowed_mask[y]:
                continue
            dist[y] = dx + 1
            q.append(y)

    return farthest, dist, maxd


def farthest_first_seeds(indptr, indices, candidates, k: int, rng: random.Random):
    """
    Select k seeds that are far apart (approx) within the induced graph on candidates.
    Uses a handful of BFS passes.

    candidates: list/np array of node indices in the giant component.
    """
    if k <= 0 or len(candidates) == 0:
        return []

    n = indptr.shape[0] - 1
    allowed = np.zeros(n, dtype=bool)
    allowed[np.asarray(candidates, dtype=np.int32)] = True

    r0 = int(rng.choice(candidates))
    a, _, _ = bfs_farthest(indptr, indices, r0, allowed_mask=allowed)
    b, dist_a, _ = bfs_farthest(indptr, indices, a, allowed_mask=allowed)

    seeds = [a, b] if a != b else [a]
    min_dist = dist_a.copy()
    if len(seeds) > 1:
        _, dist_b, _ = bfs_farthest(indptr, indices, b, allowed_mask=allowed)
        # keep min distance to any selected seed
        min_dist = np.minimum(min_dist, dist_b)

    # farthest-first iterations
    while len(seeds) < k:
        # pick argmax of min_dist over allowed nodes
        # if graph is disconnected under allowed (shouldn't be for giant comp), -1 can happen
        cand_idx = np.asarray(candidates, dtype=np.int32)
        dvals = min_dist[cand_idx]
        # exclude unreachable
        reachable = dvals >= 0
        if not np.any(reachable):
            break
        cand_idx = cand_idx[reachable]
        dvals = dvals[reachable]
        next_seed = int(cand_idx[int(np.argmax(dvals))])

        if next_seed in seeds:
            break
        seeds.append(next_seed)

        _, dist_new, _ = bfs_farthest(indptr, indices, next_seed, allowed_mask=allowed)
        min_dist = np.minimum(min_dist, dist_new)

        # if distances saturate (all small), stop early
        if int(dvals.max()) <= 2:
            break

    return seeds[:k]


# ----------------------------
# Selection mix
# ----------------------------
def compute_components(dsu: DSU, n: int):
    roots = np.empty(n, dtype=np.int32)
    for i in range(n):
        roots[i] = dsu.find(i)
    uniq, inv = np.unique(roots, return_inverse=True)
    comp_id = inv.astype(np.int32)
    comp_sizes = np.bincount(comp_id).astype(np.int64)
    return comp_id, comp_sizes


def log_bin_deg(d: int) -> int:
    if d <= 0:
        return 0
    if d == 1:
        return 1
    if d == 2:
        return 2
    if 3 <= d <= 4:
        return 3
    if 5 <= d <= 8:
        return 4
    if 9 <= d <= 16:
        return 5
    if 17 <= d <= 32:
        return 6
    if 33 <= d <= 64:
        return 7
    if 65 <= d <= 128:
        return 8
    return 9


def pick_with_comp_cap(candidates, comp_id, used_per_comp, cap_per_comp: int, k: int):
    picked = []
    for x in candidates:
        c = int(comp_id[x])
        if used_per_comp[c] >= cap_per_comp:
            continue
        picked.append(int(x))
        used_per_comp[c] += 1
        if len(picked) >= k:
            break
    return picked


def select_sources(
    node_ids: np.ndarray,
    deg: np.ndarray,
    comp_id: np.ndarray,
    comp_sizes: np.ndarray,
    indptr,
    indices,
    max_selected: int,
    seed: int,
):
    rng = random.Random(seed)
    n = node_ids.shape[0]

    # Buckets
    k_far = max(4, min(8, max_selected // 8))     # default ~8 for 64
    k_hubs = max(1, int(round(max_selected * 0.25)))
    k_fringe = max(1, int(round(max_selected * 0.25)))
    k_rand = max(1, int(round(max_selected * 0.10)))
    k_mid = max_selected - (k_far + k_hubs + k_fringe + k_rand)
    if k_mid < 0:
        k_mid = 0

    # Component caps
    overall_cap = max(2, int(math.ceil(max_selected / 16)))  # 64 -> 4
    comp_count = int(comp_sizes.shape[0])
    used = np.zeros(comp_count, dtype=np.int32)

    selected = []
    selected_set = set()

    deg64 = deg.astype(np.int64)

    # Identify giant component
    giant_comp = int(np.argmax(comp_sizes)) if comp_sizes.size else 0
    giant_nodes = np.where(comp_id == giant_comp)[0].tolist()

    # --- Far-apart seeds inside giant comp ---
    far_seeds = farthest_first_seeds(indptr, indices, giant_nodes, k_far, rng)
    for x in far_seeds:
        if x in selected_set:
            continue
        c = int(comp_id[x])
        if used[c] >= overall_cap:
            continue
        selected.append(int(x))
        selected_set.add(int(x))
        used[c] += 1

    # --- Hubs: top degrees, but component-capped ---
    sorted_by_deg = np.argsort(-deg64)
    hubs_candidates = sorted_by_deg.tolist()

    hubs_picked = pick_with_comp_cap(hubs_candidates, comp_id, used, overall_cap, k_hubs)
    for x in hubs_picked:
        if x not in selected_set:
            selected.append(int(x))
            selected_set.add(int(x))

    # --- Fringe: degree 0/1/2 first, component-capped ---
    deg0 = np.where(deg64 == 0)[0].tolist()
    deg1 = np.where(deg64 == 1)[0].tolist()
    deg2 = np.where(deg64 == 2)[0].tolist()
    rng.shuffle(deg0)
    rng.shuffle(deg1)
    rng.shuffle(deg2)
    fringe_candidates = deg0 + deg1 + deg2
    fringe_picked = pick_with_comp_cap(fringe_candidates, comp_id, used, overall_cap, k_fringe)
    for x in fringe_picked:
        if x not in selected_set:
            selected.append(int(x))
            selected_set.add(int(x))

    # --- Mid-degree: log-bin stratified, component-capped ---
    remaining = [i for i in range(n) if i not in selected_set]
    rng.shuffle(remaining)

    bins = defaultdict(list)
    for x in remaining:
        d = int(deg64[x])
        # skip very low (already fringe) and very high (already hubs) to focus mid
        if d <= 2:
            continue
        bins[log_bin_deg(d)].append(x)

    bin_keys = sorted(bins.keys())
    bin_pops = np.array([len(bins[k]) for k in bin_keys], dtype=np.float64)
    if k_mid > 0 and bin_pops.sum() > 0:
        raw = (bin_pops / bin_pops.sum()) * k_mid
        alloc = np.floor(raw).astype(int)
        leftover = k_mid - int(alloc.sum())

        # ensure coverage
        for i, k in enumerate(bin_keys):
            if leftover <= 0:
                break
            if bin_pops[i] > 0 and alloc[i] == 0:
                alloc[i] += 1
                leftover -= 1

        # distribute leftovers by frac
        if leftover > 0:
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)
            for idx in order:
                if leftover <= 0:
                    break
                alloc[idx] += 1
                leftover -= 1

        for i, k in enumerate(bin_keys):
            need = int(alloc[i])
            if need <= 0:
                continue
            cand = bins[k]
            rng.shuffle(cand)
            picked = pick_with_comp_cap(cand, comp_id, used, overall_cap, need)
            for x in picked:
                if x not in selected_set:
                    selected.append(int(x))
                    selected_set.add(int(x))

    # --- Random fill (component-capped) ---
    if len(selected) < max_selected:
        remaining = [i for i in range(n) if i not in selected_set]
        rng.shuffle(remaining)
        fill = pick_with_comp_cap(remaining, comp_id, used, overall_cap, k_rand)
        for x in fill:
            if x not in selected_set:
                selected.append(int(x))
                selected_set.add(int(x))

    # --- Final fill (relax cap if needed) ---
    if len(selected) < max_selected:
        remaining = [i for i in range(n) if i not in selected_set]
        rng.shuffle(remaining)
        for x in remaining:
            selected.append(int(x))
            selected_set.add(int(x))
            if len(selected) >= max_selected:
                break

    return selected[:max_selected]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=False, default='twitch_gamers/large_twitch_features.csv' , help="Path to large_twitch_features.csv")
    ap.add_argument("--edges", required=False, default='twitch_gamers/large_twitch_edges.csv', help="Path to large_twitch_edges.csv")
    ap.add_argument("--max_selected", type=int, default=2048, help="MAX_SELECTED")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--out_dir", default="node_selection/twitch_gamers/", help="Output directory")
    args = ap.parse_args()

    MAX_SELECTED = int(args.max_selected)

    print(f"[1/5] Loading nodes from features: {args.features}")
    node_ids, id_to_idx = read_feature_node_ids(args.features)
    n = node_ids.shape[0]
    print(f"  nodes: {n}")

    print(f"[2/5] Streaming edges for degree + components: {args.edges}")
    u, v, deg, dsu = read_edges_degree_components(args.edges, id_to_idx, n)
    m = u.shape[0]
    print(f"  edges read (mapped to known nodes): {m}")
    print(f"  degree stats: min={int(deg.min())}, mean={float(deg.mean()):.2f}, max={int(deg.max())}")

    print("[3/5] Computing connected components")
    comp_id, comp_sizes = compute_components(dsu, n)
    num_comps = int(comp_sizes.shape[0])
    giant = int(np.argmax(comp_sizes)) if num_comps else 0
    print(f"  components: {num_comps}")
    if num_comps:
        print(f"  giant component size: {int(comp_sizes[giant])}")

    print("[4/5] Building CSR adjacency (undirected)")
    indptr, indices = build_csr_undirected(n, u, v)
    print(f"  adjacency entries: {int(indices.shape[0])}")

    print(f"[5/5] Selecting {MAX_SELECTED} representative sources")
    selected_idx = select_sources(
        node_ids=node_ids,
        deg=deg,
        comp_id=comp_id,
        comp_sizes=comp_sizes,
        indptr=indptr,
        indices=indices,
        max_selected=MAX_SELECTED,
        seed=int(args.seed),
    )
    selected_ids = [int(node_ids[i]) for i in selected_idx]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"selected_nodes_{MAX_SELECTED}.txt")
    json_path = os.path.join(out_dir, f"selected_nodes_{MAX_SELECTED}.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for nid in selected_ids:
            f.write(f"{nid}\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, indent=2)

    # Report
    sel_deg = deg[np.asarray(selected_idx, dtype=np.int32)]
    sel_comp = comp_id[np.asarray(selected_idx, dtype=np.int32)]
    uniq_comps = int(np.unique(sel_comp).shape[0])
    print("Selection summary:")
    print(f"  unique components covered: {uniq_comps}/{int(comp_sizes.shape[0])}")
    print(f"  selected degree stats: min={int(sel_deg.min())}, p50={int(np.median(sel_deg))}, max={int(sel_deg.max())}")

    # small breakdown counts
    print("  bucket-ish sanity (counts among selected):")
    print(f"    deg==0: {int(np.sum(sel_deg == 0))}")
    print(f"    deg==1: {int(np.sum(sel_deg == 1))}")
    print(f"    deg==2: {int(np.sum(sel_deg == 2))}")
    print(f"    deg>=64: {int(np.sum(sel_deg >= 64))}")

    print(f"Wrote:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    main()