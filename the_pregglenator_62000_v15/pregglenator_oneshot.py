"""
pregglenator 62000 v18 - one-shot partitioning with "unlimited runtime" upgrades:

Upgrades vs v17:
  1) FM refinement -> Simulated Annealing (SA) refinement.
     - Accept improving moves always
     - Accept worsening moves with prob exp(-Δ/T)
     - Track best_global_state and revert to it at the end of refinement

  2) Hard capacity walls -> Soft slack constraints with quadratic penalties.
     - Allow temporary overflow up to slack_factor * cap (hard ceiling)
     - Add quadratic penalties for exceeding strict caps (worker and machine)

  3) Affinity coarsening (pre-contraction) before initial METIS.
     - Contract very heavy edges (>= coarsen_percentile) into super-nodes
     - Partition coarsened graph with METIS
     - Uncoarsen to node-level assignment before annealing + alternating steps

Still includes the v17 alternation framework:
  (A) SA node moves given part->machine mapping
  (B) part->machine remapping via capacity-aware swap local search
Repeat for a few rounds.

Install:
  conda install -c conda-forge pymetis
"""

import argparse
import json
import math
from collections import defaultdict, deque
import random

import pymetis


# --------------------------
# IO
# --------------------------

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


def load_comm_edgelist(path, directed=True):
    """
    Edge list format:
      src dst weight
    Lines starting with # are ignored.
    """
    comm = defaultdict(dict)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            s = int(parts[0])
            t = int(parts[1])
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            comm[s][t] = comm[s].get(t, 0.0) + w
            if not directed:
                comm[t][s] = comm[t].get(s, 0.0) + w
    return dict(comm)


def infer_num_nodes(comm, explicit_n=None):
    if explicit_n is not None:
        return explicit_n
    mx = -1
    for s, nbrs in comm.items():
        mx = max(mx, s)
        for t in nbrs.keys():
            mx = max(mx, t)
    return mx + 1


# --------------------------
# Graph building
# --------------------------

def symmetrize_to_undirected(comm, n):
    """
    Returns undirected adjacency with weights:
      und[u][v] = comm[u][v] + comm[v][u], u != v
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
            w_vu = adj[v].get(u, 0.0)
            w = w_uv + w_vu
            if w <= 0:
                continue
            und[u][v] = w
            und[v][u] = w
    return und


def build_pymetis_inputs(und_adj):
    """
    pymetis expects:
      xadj: offsets
      adjncy: neighbor ids
      eweights: edge weights aligned with adjncy
    """
    n = len(und_adj)
    xadj = [0]
    adjncy = []
    eweights = []
    for u in range(n):
        items = sorted(und_adj[u].items(), key=lambda kv: kv[0])
        for v, w in items:
            adjncy.append(v)
            eweights.append(int(round(w)))  # pymetis wants ints
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights


def metis_partition(und_adj, nparts, vweights=None, seed=42):
    """
    vweights: list[int] length n (optional). If pymetis build doesn't support it, we fallback.
    """
    xadj, adjncy, eweights = build_pymetis_inputs(und_adj)
    try:
        if vweights is None:
            _, parts = pymetis.part_graph(
                nparts,
                xadj=xadj,
                adjncy=adjncy,
                eweights=eweights,
            )
        else:
            _, parts = pymetis.part_graph(
                nparts,
                xadj=xadj,
                adjncy=adjncy,
                eweights=eweights,
                vweights=[int(x) for x in vweights],
            )
    except TypeError:
        # Some pymetis builds don't accept vweights; fall back.
        _, parts = pymetis.part_graph(
            nparts,
            xadj=xadj,
            adjncy=adjncy,
            eweights=eweights,
        )
    return list(parts)


# --------------------------
# Affinity Coarsening (pre-contraction)
# --------------------------

def _percentile_threshold(values, pct):
    """
    pct in [0, 100]. Returns threshold such that approximately pct% of values are <= threshold.
    """
    if not values:
        return float("inf")
    xs = sorted(values)
    k = int(round((pct / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


def affinity_coarsen_pairs(
    und_adj,
    percentile=95.0,
    max_pair_fraction=0.45,
    seed=0,
):
    """
    Contracts heavy edges into super-nodes using a greedy maximal matching.

    Returns:
      coarse_adj: list[dict] undirected weighted adjacency on super-nodes
      super_of: list[int] mapping original node -> super-node id
      members: list[list[int]] members[super] = original nodes
      super_size: list[int] node-count per super-node
    """
    rng = random.Random(seed)
    n = len(und_adj)

    edges = []
    weights = []
    for u in range(n):
        for v, w in und_adj[u].items():
            if v <= u:
                continue
            w = float(w)
            edges.append((u, v, w))
            weights.append(w)

    if not edges:
        super_of = list(range(n))
        members = [[u] for u in range(n)]
        super_size = [1] * n
        coarse_adj = [defaultdict(float) for _ in range(n)]
        return coarse_adj, super_of, members, super_size

    thr = _percentile_threshold(weights, percentile)

    heavy = [(u, v, w) for (u, v, w) in edges if w >= thr]
    heavy.sort(key=lambda x: x[2], reverse=True)

    unmatched = [True] * n
    pairs = []

    max_pairs = int(max_pair_fraction * n)
    made = 0
    for u, v, w in heavy:
        if made >= max_pairs:
            break
        if unmatched[u] and unmatched[v]:
            unmatched[u] = False
            unmatched[v] = False
            pairs.append((u, v))
            made += 1

    # Build supernode mapping
    super_of = [-1] * n
    members = []
    for u, v in pairs:
        sid = len(members)
        members.append([u, v])
        super_of[u] = sid
        super_of[v] = sid

    for u in range(n):
        if super_of[u] == -1:
            sid = len(members)
            members.append([u])
            super_of[u] = sid

    sn = len(members)
    super_size = [len(members[s]) for s in range(sn)]

    coarse_adj = [defaultdict(float) for _ in range(sn)]
    for u in range(n):
        su = super_of[u]
        for v, w in und_adj[u].items():
            sv = super_of[v]
            if su == sv:
                continue
            coarse_adj[su][sv] += float(w)

    # symmetrize (coarse_adj currently double-counted because und_adj is symmetric; fix to symmetric dicts)
    und = [defaultdict(float) for _ in range(sn)]
    seen = set()
    for su in range(sn):
        for sv, w in coarse_adj[su].items():
            if (sv, su) in seen:
                continue
            seen.add((su, sv))
            w2 = coarse_adj[sv].get(su, 0.0)
            ww = float(w) + float(w2)
            if ww <= 0:
                continue
            und[su][sv] = ww
            und[sv][su] = ww

    return und, super_of, members, super_size


def uncoarsen_part_assignment(super_part_of, members):
    """
    Given supernode assignment, expand to node-level part_of.
    """
    n = sum(len(m) for m in members)
    part_of = [-1] * n
    for s, nodes in enumerate(members):
        p = int(super_part_of[s])
        for u in nodes:
            part_of[u] = p
    return part_of


# --------------------------
# Capacity helpers
# --------------------------

def compute_boundary_nodes(und_adj, part_of):
    boundary = set()
    for u in range(len(und_adj)):
        pu = part_of[u]
        for v in und_adj[u].keys():
            if part_of[v] != pu:
                boundary.add(u)
                break
    return boundary


def move_cost_delta_cut_only(und_adj, u, src_part, dst_part, part_of):
    """
    Standard cut objective delta for moving u from src_part to dst_part.
    Negative means improvement.
    """
    delta = 0.0
    for v, w in und_adj[u].items():
        pv = part_of[v]
        before_cut = 1 if pv != src_part else 0
        after_cut = 1 if pv != dst_part else 0
        delta += w * (after_cut - before_cut)
    return delta


def repair_capacity_per_part_strict(und_adj, part_of, capacity, nparts):
    """
    Strict: enforce max size per part <= capacity, via greedy boundary moves under cut-only.
    Used as a final cleanup to satisfy strict execution constraints.
    """
    n = len(part_of)
    sizes = [0] * nparts
    for p in part_of:
        sizes[p] += 1

    overloaded = deque([p for p in range(nparts) if sizes[p] > capacity])
    if not overloaded:
        return part_of

    boundary = compute_boundary_nodes(und_adj, part_of)
    underfull = set([p for p in range(nparts) if sizes[p] < capacity])

    if not underfull and overloaded:
        raise RuntimeError("Strict worker capacity infeasible: all parts at capacity but at least one overloaded.")

    while overloaded:
        p_over = overloaded.popleft()
        if sizes[p_over] <= capacity:
            continue

        candidates = []
        for u in list(boundary):
            if part_of[u] != p_over:
                continue
            for p_to in underfull:
                d = move_cost_delta_cut_only(und_adj, u, p_over, p_to, part_of)
                candidates.append((d, u, p_to))

        if not candidates:
            for u in range(n):
                if part_of[u] != p_over:
                    continue
                for p_to in underfull:
                    d = move_cost_delta_cut_only(und_adj, u, p_over, p_to, part_of)
                    candidates.append((d, u, p_to))

        if not candidates:
            raise RuntimeError("Strict repair failed: no candidates to move from an overloaded part.")

        candidates.sort(key=lambda x: x[0])
        _, u_best, p_to_best = candidates[0]

        part_of[u_best] = p_to_best
        sizes[p_over] -= 1
        sizes[p_to_best] += 1

        if sizes[p_to_best] >= capacity:
            underfull.discard(p_to_best)

        boundary.add(u_best)
        for v in und_adj[u_best].keys():
            boundary.add(v)

        if sizes[p_over] > capacity:
            overloaded.append(p_over)

        if not underfull and any(s > capacity for s in sizes):
            raise RuntimeError("Strict capacity infeasible during repair: not enough slack to fix overload.")

    return part_of


def compute_part_sizes(part_of, nparts):
    sz = [0] * nparts
    for p in part_of:
        sz[p] += 1
    return sz


def compute_machine_loads(part_sizes, part_machine, num_machines):
    loads = [0] * num_machines
    for p, sz in enumerate(part_sizes):
        m = part_machine[p]
        loads[m] += sz
    return loads


# --------------------------
# Part graph (traffic between parts)
# --------------------------

def build_part_traffic(und_adj, part_of, nparts):
    """
    traffic[p][q] = sum of weights of edges crossing between part p and part q (undirected, counted once)
    """
    traffic = [defaultdict(float) for _ in range(nparts)]
    seen = set()
    n = len(und_adj)
    for u in range(n):
        pu = part_of[u]
        for v, w in und_adj[u].items():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            pv = part_of[v]
            if pu == pv:
                continue
            traffic[pu][pv] += float(w)
            traffic[pv][pu] += float(w)
    return traffic


# --------------------------
# Objective + penalties (soft constraints)
# --------------------------

def _quad_overflow_penalty(x, cap, lam):
    """
    Quadratic penalty on overflow beyond strict cap.
    """
    over = x - cap
    if over <= 0:
        return 0.0
    return lam * float(over * over)


def compute_assignment_cost_twolevel_soft(
    und_adj,
    part_of,
    part_machine,
    part_sizes,
    alpha,
    beta,
    nodes_per_worker,
    nodes_per_machine,
    lambda_worker,
    lambda_machine,
):
    """
    Cost = two-level edge cost + soft quadratic penalties for worker and machine overflow beyond strict caps.
    Edges counted once (undirected).
    """
    # edge cost
    total = 0.0
    seen = set()
    n = len(und_adj)
    for u in range(n):
        pu = part_of[u]
        mu = part_machine[pu]
        for v, w in und_adj[u].items():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            pv = part_of[v]
            if pu == pv:
                continue
            mv = part_machine[pv]
            total += (beta * w) if (mu == mv) else (alpha * w)

    # penalties
    # worker
    for sz in part_sizes:
        total += _quad_overflow_penalty(sz, nodes_per_worker, lambda_worker)

    # machine
    num_machines = max(part_machine) + 1 if part_machine else 0
    machine_load = [0] * num_machines
    for p, sz in enumerate(part_sizes):
        machine_load[part_machine[p]] += sz
    for ml in machine_load:
        total += _quad_overflow_penalty(ml, nodes_per_machine, lambda_machine)

    return total


def compute_cost_breakdown_raw_undirected(und_adj, part_of, part_machine):
    """
    Returns (intra_machine_cut_weight, inter_machine_cut_weight) with raw undirected weights (no alpha/beta).
    """
    intra = 0.0
    inter = 0.0
    seen = set()
    n = len(und_adj)
    for u in range(n):
        pu = part_of[u]
        mu = part_machine[pu]
        for v, w in und_adj[u].items():
            if (v, u) in seen:
                continue
            seen.add((u, v))
            pv = part_of[v]
            if pu == pv:
                continue
            mv = part_machine[pv]
            if mu == mv:
                intra += float(w)
            else:
                inter += float(w)
    return intra, inter


# --------------------------
# SA refinement (node moves) under two-level + soft penalties
# --------------------------

def move_delta_twolevel_soft(
    und_adj,
    u,
    src_part,
    dst_part,
    part_of,
    part_machine,
    part_sizes,
    machine_loads,
    alpha,
    beta,
    nodes_per_worker,
    nodes_per_machine,
    lambda_worker,
    lambda_machine,
):
    """
    Δ = edge-cost change + penalty change (worker + machine).
    """
    # edge delta
    delta = 0.0
    m_src = part_machine[src_part]
    m_dst = part_machine[dst_part]

    for v, w in und_adj[u].items():
        pv = part_of[v]

        if pv == src_part:
            before = 0.0
        else:
            before = (beta * w) if (part_machine[pv] == m_src) else (alpha * w)

        if pv == dst_part:
            after = 0.0
        else:
            after = (beta * w) if (part_machine[pv] == m_dst) else (alpha * w)

        delta += (after - before)

    # penalty delta: worker overflow changes for src and dst only
    src_sz = part_sizes[src_part]
    dst_sz = part_sizes[dst_part]

    before_pw = _quad_overflow_penalty(src_sz, nodes_per_worker, lambda_worker) + _quad_overflow_penalty(dst_sz, nodes_per_worker, lambda_worker)
    after_pw = _quad_overflow_penalty(src_sz - 1, nodes_per_worker, lambda_worker) + _quad_overflow_penalty(dst_sz + 1, nodes_per_worker, lambda_worker)
    delta += (after_pw - before_pw)

    # penalty delta: machine overflow changes for m_src and m_dst (if different)
    if m_src == m_dst:
        # only worker penalties matter; machine load unchanged
        return delta

    ml_src = machine_loads[m_src]
    ml_dst = machine_loads[m_dst]

    before_pm = _quad_overflow_penalty(ml_src, nodes_per_machine, lambda_machine) + _quad_overflow_penalty(ml_dst, nodes_per_machine, lambda_machine)
    after_pm = _quad_overflow_penalty(ml_src - 1, nodes_per_machine, lambda_machine) + _quad_overflow_penalty(ml_dst + 1, nodes_per_machine, lambda_machine)
    delta += (after_pm - before_pm)

    return delta


def refine_sa_twolevel_soft(
    und_adj,
    part_of,
    part_machine,
    num_machines,
    workers_per_machine,
    nodes_per_machine,
    nodes_per_worker,
    alpha,
    beta,
    slack_factor,
    lambda_worker,
    lambda_machine,
    seed=0,
    passes=8,
    steps_per_pass=None,
    T0=1.0,
    T_decay=0.85,
    candidate_parts_cap=64,
):
    """
    Simulated annealing refinement.

    - Builds boundary node pool.
    - Repeatedly proposes moves of a random boundary node to a candidate destination part.
    - Accepts:
        Δ < 0 always
        Δ > 0 with prob exp(-Δ/T)
    - Uses soft penalties for overflow beyond strict caps.
    - Hard ceiling: do not allow sizes to exceed ceil(slack_factor * cap) for worker or machine.

    Tracks best_global_state by objective+penalties and returns that best.
    """
    rng = random.Random(seed)
    n = len(part_of)
    nparts = num_machines * workers_per_machine

    if steps_per_pass is None:
        steps_per_pass = max(200, 5 * n)

    # sizes / loads
    part_sizes = [0] * nparts
    for p in part_of:
        part_sizes[p] += 1

    machine_loads = [0] * num_machines
    for p in range(nparts):
        machine_loads[part_machine[p]] += part_sizes[p]

    # hard ceilings
    worker_ceiling = int(math.ceil(slack_factor * nodes_per_worker))
    machine_ceiling = int(math.ceil(slack_factor * nodes_per_machine))

    def is_boundary(u):
        pu = part_of[u]
        for v in und_adj[u].keys():
            if part_of[v] != pu:
                return True
        return False

    boundary_nodes = [u for u in range(n) if is_boundary(u)]
    if not boundary_nodes:
        return part_of

    # best-global tracking
    best_part_of = list(part_of)
    best_part_sizes = list(part_sizes)
    best_machine_loads = list(machine_loads)

    best_cost = compute_assignment_cost_twolevel_soft(
        und_adj=und_adj,
        part_of=part_of,
        part_machine=part_machine,
        part_sizes=part_sizes,
        alpha=alpha,
        beta=beta,
        nodes_per_worker=nodes_per_worker,
        nodes_per_machine=nodes_per_machine,
        lambda_worker=lambda_worker,
        lambda_machine=lambda_machine,
    )

    T = float(T0)

    for _p in range(passes):
        for _step in range(steps_per_pass):
            u = boundary_nodes[rng.randrange(0, len(boundary_nodes))]
            src = part_of[u]

            # candidate dst parts from neighbor parts + a few random escapes
            cand_parts = set()
            for v in und_adj[u].keys():
                pv = part_of[v]
                if pv != src:
                    cand_parts.add(pv)
                if len(cand_parts) >= candidate_parts_cap:
                    break
            for _ in range(6):
                cand_parts.add(rng.randrange(0, nparts))

            if not cand_parts:
                continue

            dst = None
            # sample a few candidates and pick one at random among them (keeps SA stochastic)
            cand_list = list(cand_parts)
            dst = cand_list[rng.randrange(0, len(cand_list))]
            if dst == src:
                continue

            # hard ceiling guardrails
            if part_sizes[dst] + 1 > worker_ceiling:
                continue
            m_src = part_machine[src]
            m_dst = part_machine[dst]
            if m_src != m_dst and (machine_loads[m_dst] + 1 > machine_ceiling):
                continue

            delta = move_delta_twolevel_soft(
                und_adj=und_adj,
                u=u,
                src_part=src,
                dst_part=dst,
                part_of=part_of,
                part_machine=part_machine,
                part_sizes=part_sizes,
                machine_loads=machine_loads,
                alpha=alpha,
                beta=beta,
                nodes_per_worker=nodes_per_worker,
                nodes_per_machine=nodes_per_machine,
                lambda_worker=lambda_worker,
                lambda_machine=lambda_machine,
            )

            accept = False
            if delta <= 0.0:
                accept = True
            else:
                # exp(-Δ/T)
                if T > 1e-12:
                    p_acc = math.exp(-float(delta) / float(T))
                    if rng.random() < p_acc:
                        accept = True

            if not accept:
                continue

            # apply move
            part_of[u] = dst
            part_sizes[src] -= 1
            part_sizes[dst] += 1
            if m_src != m_dst:
                machine_loads[m_src] -= 1
                machine_loads[m_dst] += 1

            # update boundary status opportunistically (cheap: keep pool static; SA still works)
            # Note: static boundary pool is fine for quality; it just wastes some steps.

            # best-global update (compute full cost occasionally for stability)
            # Here, do it every accepted move for simplicity (unlimited runtime assumption).
            cur_cost = compute_assignment_cost_twolevel_soft(
                und_adj=und_adj,
                part_of=part_of,
                part_machine=part_machine,
                part_sizes=part_sizes,
                alpha=alpha,
                beta=beta,
                nodes_per_worker=nodes_per_worker,
                nodes_per_machine=nodes_per_machine,
                lambda_worker=lambda_worker,
                lambda_machine=lambda_machine,
            )
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_part_of = list(part_of)
                best_part_sizes = list(part_sizes)
                best_machine_loads = list(machine_loads)

        T *= float(T_decay)

    # revert to best state seen
    for i in range(n):
        part_of[i] = best_part_of[i]

    return part_of


# --------------------------
# Part->machine remapping optimization (swap search, capacity-aware)
# --------------------------

def _traffic_between(traffic, a, b):
    if a == b:
        return 0.0
    return float(traffic[a].get(b, 0.0))


def improve_mapping_by_swaps(
    part_traffic,
    part_machine,
    part_sizes,
    num_machines,
    workers_per_machine,
    nodes_per_machine,
    seed=0,
    max_rounds=10,
    max_swaps_per_round=2500,
):
    """
    Local search over part->machine mapping:
      - machines have fixed number of parts (workers_per_machine)
      - swap one part from machine A with one from machine B
      - accept if it increases within-machine traffic sum
      - enforce strict machine node-load capacity after swap (hard)

    This step is not SA; it's steepest ascent on within-machine affinity.
    """
    rng = random.Random(seed)
    nparts = len(part_machine)

    machine_parts = [[] for _ in range(num_machines)]
    for p in range(nparts):
        m = part_machine[p]
        machine_parts[m].append(p)

    machine_load = [0] * num_machines
    for m in range(num_machines):
        s = 0
        for p in machine_parts[m]:
            s += part_sizes[p]
        machine_load[m] = s

    def within_contrib(p, m):
        s = 0.0
        for x in machine_parts[m]:
            if x == p:
                continue
            s += _traffic_between(part_traffic, p, x)
        return s

    for _ in range(max_rounds):
        improved = False
        swaps_done = 0

        ms = list(range(num_machines))
        rng.shuffle(ms)

        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if swaps_done >= max_swaps_per_round:
                    break

                ma = ms[i]
                mb = ms[j]
                if not machine_parts[ma] or not machine_parts[mb]:
                    continue

                best_gain = 0.0
                best_pair = None

                # brute force swap search between these machines
                for pa in machine_parts[ma]:
                    ca = within_contrib(pa, ma)
                    cb = within_contrib(pa, mb)
                    for pb in machine_parts[mb]:
                        da = within_contrib(pb, mb)
                        db = within_contrib(pb, ma)

                        new_load_a = machine_load[ma] - part_sizes[pa] + part_sizes[pb]
                        new_load_b = machine_load[mb] - part_sizes[pb] + part_sizes[pa]
                        if new_load_a > nodes_per_machine or new_load_b > nodes_per_machine:
                            continue

                        gain = (cb + db) - (ca + da)
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (pa, pb, new_load_a, new_load_b)

                if best_pair is not None and best_gain > 1e-9:
                    pa, pb, new_load_a, new_load_b = best_pair

                    machine_parts[ma].remove(pa)
                    machine_parts[mb].remove(pb)
                    machine_parts[ma].append(pb)
                    machine_parts[mb].append(pa)

                    part_machine[pa] = mb
                    part_machine[pb] = ma

                    machine_load[ma] = new_load_a
                    machine_load[mb] = new_load_b

                    swaps_done += 1
                    improved = True

            if swaps_done >= max_swaps_per_round:
                break

        if not improved:
            break

    return part_machine


# --------------------------
# Initial grouping of parts into machines (capacity-aware greedy)
# --------------------------

def greedy_group_parts_into_machines(part_traffic, num_machines, workers_per_machine, part_sizes, nodes_per_machine, seed=42):
    """
    Assign each part -> machine, with exactly workers_per_machine parts per machine.

    Heuristic objective: maximize within-machine traffic, subject to machine load cap.
    """
    rng = random.Random(seed)
    nparts = len(part_traffic)

    total_degree = []
    for p in range(nparts):
        deg = 0.0
        for _, w in part_traffic[p].items():
            deg += float(w)
        total_degree.append(deg)

    unassigned = set(range(nparts))
    part_machine = [-1] * nparts
    machine_parts = [[] for _ in range(num_machines)]
    machine_load = [0] * num_machines

    parts_sorted = sorted(range(nparts), key=lambda p: (total_degree[p], rng.random()), reverse=True)

    for m in range(num_machines):
        if not unassigned:
            break

        seed_part = None
        for p in parts_sorted:
            if p in unassigned and machine_load[m] + part_sizes[p] <= nodes_per_machine:
                seed_part = p
                break
        if seed_part is None:
            for p in parts_sorted:
                if p in unassigned:
                    seed_part = p
                    break
        if seed_part is None:
            break

        group = [seed_part]
        unassigned.remove(seed_part)
        part_machine[seed_part] = m
        machine_load[m] += part_sizes[seed_part]

        while len(group) < workers_per_machine and unassigned:
            best_p = None
            best_score = -1.0
            for cand in unassigned:
                if machine_load[m] + part_sizes[cand] > nodes_per_machine:
                    continue
                score = 0.0
                for gp in group:
                    score += float(part_traffic[cand].get(gp, 0.0))
                if score > best_score:
                    best_score = score
                    best_p = cand

            if best_p is None:
                break

            group.append(best_p)
            unassigned.remove(best_p)
            part_machine[best_p] = m
            machine_load[m] += part_sizes[best_p]

        machine_parts[m] = group

    if unassigned:
        leftovers = sorted(list(unassigned))
        for p in leftovers:
            placed = False
            ms = sorted(range(num_machines), key=lambda mm: machine_load[mm])
            for m in ms:
                if len(machine_parts[m]) >= workers_per_machine:
                    continue
                if machine_load[m] + part_sizes[p] <= nodes_per_machine:
                    part_machine[p] = m
                    machine_parts[m].append(p)
                    machine_load[m] += part_sizes[p]
                    placed = True
                    break
            if not placed:
                m = min(range(num_machines), key=lambda mm: machine_load[mm])
                part_machine[p] = m
                machine_parts[m].append(p)
                machine_load[m] += part_sizes[p]

    return part_machine, machine_parts


# --------------------------
# One-shot pipeline (alternating with SA + soft slack + coarsening)
# --------------------------

def partition_one_shot_sa_alternating(
    und_adj,
    num_machines,
    nodes_per_machine,
    nodes_per_worker,
    alpha,
    beta,
    seed=42,
    # Alternation knobs
    alt_rounds=8,
    remap_rounds=10,
    # SA knobs
    sa_passes=8,
    sa_steps_per_pass=None,
    sa_T0=None,
    sa_T_decay=0.85,
    # Soft constraints
    slack_factor=1.05,
    lambda_worker=10.0,
    lambda_machine=10.0,
    # Coarsening knobs
    coarsen_percentile=95.0,
    coarsen_max_pair_fraction=0.45,
    # Final strictness
    final_strict_repair=True,
):
    n = len(und_adj)
    if num_machines * nodes_per_machine < n:
        raise RuntimeError("Infeasible: NUM_MACHINES * NODES_PER_MACHINE < total nodes.")

    workers_per_machine = int(math.ceil(nodes_per_machine / float(nodes_per_worker)))
    if workers_per_machine <= 0:
        workers_per_machine = 1
    total_workers = num_machines * workers_per_machine

    # --- (3) Affinity coarsening
    coarse_adj, super_of, members, super_size = affinity_coarsen_pairs(
        und_adj,
        percentile=coarsen_percentile,
        max_pair_fraction=coarsen_max_pair_fraction,
        seed=seed + 11,
    )

    # Partition coarsened graph with METIS
    # Use vweights = super_size to encourage balancing by node mass.
    super_part = metis_partition(coarse_adj, total_workers, vweights=super_size, seed=seed + 21)

    # Uncoarsen to node-level parts
    part_of = uncoarsen_part_assignment(super_part, members)

    # Ensure all nodes got assigned
    if any(p < 0 or p >= total_workers for p in part_of):
        raise RuntimeError("Uncoarsen produced invalid part ids.")

    # (optional) strict repair to get near-feasible before soft search (keeps things sane)
    # Note: coarsening can produce temporary overload; strict repair helps.
    part_of = repair_capacity_per_part_strict(und_adj, part_of, nodes_per_worker, total_workers)

    # Initial mapping parts->machines
    part_sizes = compute_part_sizes(part_of, total_workers)
    part_traffic = build_part_traffic(und_adj, part_of, total_workers)
    part_machine, machine_parts = greedy_group_parts_into_machines(
        part_traffic,
        num_machines,
        workers_per_machine,
        part_sizes=part_sizes,
        nodes_per_machine=nodes_per_machine,
        seed=seed + 101,
    )

    # SA temperature heuristic (if not provided)
    if sa_T0 is None:
        # Scale T0 to a fraction of typical move delta magnitude.
        # Use mean edge weight * (alpha) / 10 as a crude baseline.
        # Unlimited runtime means this doesn't need to be perfect.
        wsum = 0.0
        wcnt = 0
        for u in range(n):
            for v, w in und_adj[u].items():
                if v > u:
                    wsum += float(w)
                    wcnt += 1
        mean_w = (wsum / float(wcnt)) if wcnt > 0 else 1.0
        sa_T0 = max(1.0, 0.1 * float(alpha) * mean_w)

    # Alternation loop
    for r in range(alt_rounds):
        # (1) SA node refinement with soft penalties
        part_of = refine_sa_twolevel_soft(
            und_adj=und_adj,
            part_of=part_of,
            part_machine=part_machine,
            num_machines=num_machines,
            workers_per_machine=workers_per_machine,
            nodes_per_machine=nodes_per_machine,
            nodes_per_worker=nodes_per_worker,
            alpha=alpha,
            beta=beta,
            slack_factor=slack_factor,
            lambda_worker=lambda_worker,
            lambda_machine=lambda_machine,
            seed=seed + 2000 + r,
            passes=sa_passes,
            steps_per_pass=sa_steps_per_pass,
            T0=sa_T0,
            T_decay=sa_T_decay,
        )

        # (2) rebuild part graph
        part_sizes = compute_part_sizes(part_of, total_workers)
        part_traffic = build_part_traffic(und_adj, part_of, total_workers)

        # (3) remap parts->machines by swaps (strict machine cap)
        part_machine = improve_mapping_by_swaps(
            part_traffic=part_traffic,
            part_machine=part_machine,
            part_sizes=part_sizes,
            num_machines=num_machines,
            workers_per_machine=workers_per_machine,
            nodes_per_machine=nodes_per_machine,
            seed=seed + 3000 + r,
            max_rounds=remap_rounds,
        )

    # Optional final strict repair for worker caps (execution requirement)
    if final_strict_repair:
        part_of = repair_capacity_per_part_strict(und_adj, part_of, nodes_per_worker, total_workers)

    # Assign local worker ids within each machine for output
    machine_parts = [[] for _ in range(num_machines)]
    for p in range(total_workers):
        machine_parts[part_machine[p]].append(p)

    part_worker_local = [-1] * total_workers
    for m in range(num_machines):
        parts = sorted(machine_parts[m])
        for i, p in enumerate(parts):
            part_worker_local[p] = i

    machine_of = [-1] * n
    worker_of = [-1] * n
    for u in range(n):
        p = part_of[u]
        machine_of[u] = part_machine[p]
        worker_of[u] = part_worker_local[p]

    worker_count_per_machine = [workers_per_machine for _ in range(num_machines)]
    return (
        machine_of,
        worker_of,
        worker_count_per_machine,
        workers_per_machine,
        total_workers,
        part_of,
        part_machine,
    )


# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to comm traces (json dict-of-dicts or edgelist)")
    ap.add_argument("--format", choices=["json", "edgelist"], required=True)
    ap.add_argument("--directed", action="store_true", help="Treat edgelist as directed (default for edgelist)")
    ap.add_argument("--num_nodes", type=int, default=None, help="Optional explicit N. Otherwise inferred.")

    ap.add_argument("--num_machines", type=int, required=True)
    ap.add_argument("--nodes_per_machine", type=int, required=True)
    ap.add_argument("--nodes_per_worker", type=int, required=True)

    ap.add_argument("--alpha", type=float, default=50.0, help="Cross-machine penalty multiplier")
    ap.add_argument("--beta", type=float, default=1.0, help="Intra-machine cross-worker penalty multiplier")

    ap.add_argument("--seed", type=int, default=42)

    # Alternation knobs
    ap.add_argument("--alt_rounds", type=int, default=10)
    ap.add_argument("--remap_rounds", type=int, default=12)

    # SA knobs
    ap.add_argument("--sa_passes", type=int, default=10)
    ap.add_argument("--sa_steps_per_pass", type=int, default=None)  # None => auto
    ap.add_argument("--sa_T0", type=float, default=None)            # None => heuristic
    ap.add_argument("--sa_T_decay", type=float, default=0.85)

    # Soft constraints
    ap.add_argument("--slack_factor", type=float, default=1.05, help="Hard ceiling multiplier over strict caps (e.g., 1.05 = 5% slack)")
    ap.add_argument("--lambda_worker", type=float, default=10.0, help="Quadratic overflow penalty weight for worker parts")
    ap.add_argument("--lambda_machine", type=float, default=10.0, help="Quadratic overflow penalty weight for machines")

    # Coarsening
    ap.add_argument("--coarsen_percentile", type=float, default=95.0)
    ap.add_argument("--coarsen_max_pair_fraction", type=float, default=0.45)

    # Final strictness
    ap.add_argument("--final_strict_repair", action="store_true", help="Enforce strict worker caps at the very end")
    ap.add_argument("--output", required=True, help="Output JSON mapping node -> machine/worker")
    

    # Backwards-compat: old ultrasound.py passes this
    ap.add_argument(
        "--max_refine_passes",
        type=int,
        default=None,
        help="DEPRECATED alias for --sa_passes (kept for compatibility with older drivers)",
    )
    
    args = ap.parse_args()

    if args.alpha < args.beta:
        raise RuntimeError("Require alpha >= beta (cross-machine should be >= intra-machine).")
    if args.slack_factor < 1.0:
        raise RuntimeError("slack_factor must be >= 1.0")

    if args.format == "json":
        comm = load_comm_json(args.input)
    else:
        comm = load_comm_edgelist(args.input, directed=args.directed)

    n = infer_num_nodes(comm, explicit_n=args.num_nodes)
    und_adj = symmetrize_to_undirected(comm, n)

    (
        machine_of,
        worker_of,
        worker_counts,
        workers_per_machine,
        total_workers,
        part_of,
        part_machine,
    ) = partition_one_shot_sa_alternating(
        und_adj=und_adj,
        num_machines=args.num_machines,
        nodes_per_machine=args.nodes_per_machine,
        nodes_per_worker=args.nodes_per_worker,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
        alt_rounds=args.alt_rounds,
        remap_rounds=args.remap_rounds,
        sa_passes=args.sa_passes,
        sa_steps_per_pass=args.sa_steps_per_pass,
        sa_T0=args.sa_T0,
        sa_T_decay=args.sa_T_decay,
        slack_factor=args.slack_factor,
        lambda_worker=args.lambda_worker,
        lambda_machine=args.lambda_machine,
        coarsen_percentile=args.coarsen_percentile,
        coarsen_max_pair_fraction=args.coarsen_max_pair_fraction,
        final_strict_repair=args.final_strict_repair,
    )

    # Output assignment
    out = {}
    for u in range(n):
        out[str(u)] = {"machine": int(machine_of[u]), "worker": int(worker_of[u])}

    # Stats on worker parts (part_of + part_machine)
    part_sizes = compute_part_sizes(part_of, total_workers)
    intra_cut, inter_cut = compute_cost_breakdown_raw_undirected(und_adj, part_of, part_machine)

    total_cost_soft = compute_assignment_cost_twolevel_soft(
        und_adj=und_adj,
        part_of=part_of,
        part_machine=part_machine,
        part_sizes=part_sizes,
        alpha=args.alpha,
        beta=args.beta,
        nodes_per_worker=args.nodes_per_worker,
        nodes_per_machine=args.nodes_per_machine,
        lambda_worker=args.lambda_worker,
        lambda_machine=args.lambda_machine,
    )

    stats = {
        "num_nodes": n,
        "num_machines": args.num_machines,
        "nodes_per_machine": args.nodes_per_machine,
        "nodes_per_worker": args.nodes_per_worker,
        "workers_per_machine_fixed": workers_per_machine,
        "total_worker_parts": total_workers,
        "alpha": args.alpha,
        "beta": args.beta,
        "slack_factor": args.slack_factor,
        "lambda_worker": args.lambda_worker,
        "lambda_machine": args.lambda_machine,
        "coarsen_percentile": args.coarsen_percentile,
        "coarsen_max_pair_fraction": args.coarsen_max_pair_fraction,
        "alt_rounds": args.alt_rounds,
        "remap_rounds": args.remap_rounds,
        "sa_passes": args.sa_passes,
        "sa_steps_per_pass": args.sa_steps_per_pass,
        "sa_T0": args.sa_T0,
        "sa_T_decay": args.sa_T_decay,
        "final_strict_repair": bool(args.final_strict_repair),
        "intra_machine_cut_weight": intra_cut,
        "inter_machine_cut_weight": inter_cut,
        "twolevel_objective_cost_soft": total_cost_soft,
        "workers_per_machine": worker_counts,
    }

    payload = {"assignment": out, "stats": stats}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()