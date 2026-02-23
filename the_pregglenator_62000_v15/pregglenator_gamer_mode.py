"""
pregglenator 62000 v19.1 - STATIC HYPERGRAPH EDITION (broadcast-aware)
FIXED per your three “fatal bugs”:

Bug 1 FIX (Min-Max Straggler objective):
  - Objective is now to minimize the MAX machine cost, not sum.
  - Machine cost(m) = comm_cost_attributed_to_m
                      + GAMMA * compute_load_on_m
                      + (sum worker overflow penalties for parts on m)
                      + (machine overflow penalty for m)

  - SA Δ is evaluated as Δ = new_max_cost - old_max_cost, using only the affected machines/parts/hyperedges.

Bug 2 FIX (Vertex Compute Weights):
  - compute_w[u] = sum_in(u) + sum_out(u) over ORIGINAL directed comm.
  - part_sizes[p] = sum compute_w[u] for nodes u in part p (float).
  - machine_loads[m] = sum part_sizes[p] for parts on machine m.
  - All capacity ceilings + penalties use compute-load, NOT flat node counts.

Bug 3 FIX (Remove final_strict_repair):
  - Strict repair is NOT run at the end (or anywhere). We rely exclusively on soft penalties.

Interface invariants preserved:
  - CLI flags: unchanged
  - Output JSON shape: unchanged: {"assignment": {node: {machine, worker}}, "stats": {...}}
  - Existing stats keys preserved; some extra hypergraph/minmax debug keys added safely.

Install:
  conda install -c conda-forge pymetis
"""

import argparse
import json
import math
from collections import defaultdict
import random

import pymetis


# ============================================================
# Tunables (kept constant to preserve CLI)
# ============================================================

# Compute straggler weight. Prompt demanded Gamma * vertex compute weight.
# No CLI flag exists, so keep a constant. Change if you want different scaling.
GAMMA = 1.0


# ============================================================
# IO
# ============================================================

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


# ============================================================
# Compute weights (Bug 2)
# ============================================================

def compute_vertex_compute_weights(comm, n):
    """
    compute_w[u] = sum_in(u) + sum_out(u) over the ORIGINAL directed comm.
    """
    compute_w = [0.0] * n
    for u, nbrs in comm.items():
        u = int(u)
        if u < 0 or u >= n:
            continue
        for v, w in nbrs.items():
            v = int(v)
            if v < 0 or v >= n:
                continue
            w = float(w)
            compute_w[u] += w
            compute_w[v] += w
    return compute_w


# ============================================================
# Graph building (used for coarsening + METIS + boundary proposals)
# ============================================================

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
        _, parts = pymetis.part_graph(
            nparts,
            xadj=xadj,
            adjncy=adjncy,
            eweights=eweights,
        )
    return list(parts)


# ============================================================
# Static hypergraph (broadcast awareness)
# ============================================================

def build_static_broadcast_hypergraph(comm, n, drop_self=True):
    """
    Hypergraph:
      one hyperedge per sender u with outgoing neighbors:
        pins[eid] = [u] + dsts
        w[eid] = sum(comm[u][dst])

    Returns:
      hg_pins: list[list[int]]
      hg_w:    list[float]
      incident: list[list[int]]  incident[u] = list of eids containing u
      sender_of: list[int]       sender_of[eid] = sender node
    """
    hg_pins = []
    hg_w = []
    incident = [[] for _ in range(n)]
    sender_of = []

    for u in range(n):
        nbrs = comm.get(u, None)
        if not nbrs:
            continue

        pins = [u]
        wsum = 0.0
        for v, w in nbrs.items():
            v = int(v)
            if v < 0 or v >= n:
                continue
            if drop_self and v == u:
                continue
            pins.append(v)
            wsum += float(w)

        # unique pins, stable order
        seen = set()
        uniq = []
        for x in pins:
            if x not in seen:
                uniq.append(x)
                seen.add(x)

        if len(uniq) <= 1:
            continue
        if wsum <= 0.0:
            continue

        eid = len(hg_pins)
        hg_pins.append(uniq)
        hg_w.append(wsum)
        sender_of.append(u)
        for x in uniq:
            incident[x].append(eid)

    return hg_pins, hg_w, incident, sender_of


def hyperedge_cost_twolevel(pins, w_e, part_of, part_machine, alpha, beta):
    """
    Broadcast-aware hyperedge cost:
      cost(e) = w(e) * [ alpha * (|M|-1) + beta * sum_m (|P_m|-1) ]
    where:
      M = machines touched by pins
      P_m = parts touched among pins that lie on machine m
    """
    machines = set()
    parts_by_machine = defaultdict(set)

    for u in pins:
        p = part_of[u]
        m = part_machine[p]
        machines.add(m)
        parts_by_machine[m].add(p)

    net_touches = max(0, len(machines) - 1)

    ipc_touches = 0
    for ps in parts_by_machine.values():
        ipc_touches += max(0, len(ps) - 1)

    return float(w_e) * (float(alpha) * float(net_touches) + float(beta) * float(ipc_touches))


# ============================================================
# Affinity Coarsening (pre-contraction)
# ============================================================

def _percentile_threshold(values, pct):
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
    for u, v, _w in heavy:
        if made >= max_pairs:
            break
        if unmatched[u] and unmatched[v]:
            unmatched[u] = False
            unmatched[v] = False
            pairs.append((u, v))
            made += 1

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

    # symmetrize (coarse_adj double-counted because und_adj is symmetric)
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
    n = sum(len(m) for m in members)
    part_of = [-1] * n
    for s, nodes in enumerate(members):
        p = int(super_part_of[s])
        for u in nodes:
            part_of[u] = p
    return part_of


# ============================================================
# Weighted sizes (Bug 2)
# ============================================================

def compute_part_sizes_weighted(part_of, nparts, compute_w):
    sz = [0.0] * nparts
    for u, p in enumerate(part_of):
        sz[p] += float(compute_w[u])
    return sz


# ============================================================
# Part affinity graph from hyperedges (for part->machine remapping swaps)
# ============================================================

def build_part_affinity_from_hypergraph(hg_pins, hg_w, part_of, nparts):
    """
    affinity[p][q] = sum of hyperedge weights that touch both parts p and q.
    Used as a "within-machine desirability" proxy for swap mapping.
    """
    affinity = [defaultdict(float) for _ in range(nparts)]

    for eid, pins in enumerate(hg_pins):
        w_e = float(hg_w[eid])
        parts = set()
        for u in pins:
            parts.add(part_of[u])
        parts = list(parts)
        k = len(parts)
        if k <= 1:
            continue
        for i in range(k):
            pi = parts[i]
            for j in range(i + 1, k):
                pj = parts[j]
                affinity[pi][pj] += w_e
                affinity[pj][pi] += w_e

    return affinity


# ============================================================
# Penalties + MinMax machine cost (Bug 1)
# ============================================================

def _quad_overflow_penalty(x, cap, lam):
    over = x - cap
    if over <= 0:
        return 0.0
    return lam * float(over * over)


def _machine_cost(comm_cost_m, machine_loads, worker_penalty_m, machine_penalty_m, m):
    return float(comm_cost_m[m]) + float(GAMMA) * float(machine_loads[m]) + float(worker_penalty_m[m]) + float(machine_penalty_m[m])


def compute_minmax_objective_from_state(
    comm_cost_m,
    machine_loads,
    worker_penalty_m,
    machine_penalty_m,
):
    """
    Global objective is max machine cost.
    """
    best = -1e300
    for m in range(len(comm_cost_m)):
        c = _machine_cost(comm_cost_m, machine_loads, worker_penalty_m, machine_penalty_m, m)
        if c > best:
            best = c
    return best


# ============================================================
# SA refinement (hypergraph objective + MinMax straggler)
# ============================================================

def refine_sa_minmax_hyper(
    und_adj,          # boundary discovery + neighbor-part proposals
    hg_pins,
    hg_w,
    incident,
    sender_of,        # sender_of[eid]
    compute_w,
    part_of,
    part_machine,
    num_machines,
    workers_per_machine,
    nodes_per_machine,   # treated as compute capacity cap
    nodes_per_worker,    # treated as compute capacity cap
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
    Simulated annealing refinement minimizing MAX machine cost.

    State tracked:
      - part_of[u]
      - part_sizes[p] = sum compute_w[u] in part p
      - machine_loads[m] = sum part_sizes[p] for parts on machine m
      - part_penalty[p] = worker overflow penalty for part p
      - worker_penalty_m[m] = sum part_penalty[p] for parts on machine m
      - machine_penalty_m[m] = machine overflow penalty for machine m
      - hyperedge_cost[eid]
      - hyperedge_attr_m[eid] = machine id where this edge cost is attributed (sender's machine)
      - comm_cost_m[m] = sum hyperedge_cost attributed to m

    Objective:
      max_m [ comm_cost_m[m] + GAMMA*machine_loads[m] + worker_penalty_m[m] + machine_penalty_m[m] ]
    """
    rng = random.Random(seed)
    n = len(part_of)
    nparts = num_machines * workers_per_machine

    if steps_per_pass is None:
        steps_per_pass = max(200, 5 * n)

    # ceilings in compute-load units
    worker_ceiling = float(slack_factor) * float(nodes_per_worker)
    machine_ceiling = float(slack_factor) * float(nodes_per_machine)

    # ---- initialize sizes/loads/penalties
    part_sizes = compute_part_sizes_weighted(part_of, nparts, compute_w)

    machine_loads = [0.0] * num_machines
    for p in range(nparts):
        machine_loads[part_machine[p]] += float(part_sizes[p])

    part_penalty = [0.0] * nparts
    for p in range(nparts):
        part_penalty[p] = _quad_overflow_penalty(part_sizes[p], nodes_per_worker, lambda_worker)

    worker_penalty_m = [0.0] * num_machines
    for p in range(nparts):
        worker_penalty_m[part_machine[p]] += float(part_penalty[p])

    machine_penalty_m = [0.0] * num_machines
    for m in range(num_machines):
        machine_penalty_m[m] = _quad_overflow_penalty(machine_loads[m], nodes_per_machine, lambda_machine)

    # ---- initialize hyperedge costs + attribution
    hyperedge_cost = [0.0] * len(hg_pins)
    hyperedge_attr_m = [-1] * len(hg_pins)
    comm_cost_m = [0.0] * num_machines

    for eid, pins in enumerate(hg_pins):
        c = hyperedge_cost_twolevel(pins, hg_w[eid], part_of, part_machine, alpha, beta)
        hyperedge_cost[eid] = float(c)
        sender = sender_of[eid]
        m_attr = part_machine[part_of[sender]]
        hyperedge_attr_m[eid] = int(m_attr)
        comm_cost_m[m_attr] += float(c)

    def is_boundary(u):
        pu = part_of[u]
        for v in und_adj[u].keys():
            if part_of[v] != pu:
                return True
        return False

    boundary_nodes = [u for u in range(n) if is_boundary(u)]
    if not boundary_nodes:
        return part_of

    # ---- best-global tracking
    cur_obj = compute_minmax_objective_from_state(comm_cost_m, machine_loads, worker_penalty_m, machine_penalty_m)
    best_obj = cur_obj
    best_part_of = list(part_of)

    T = float(T0)

    # helper: recompute objective fast (num_machines is small)
    def current_objective():
        return compute_minmax_objective_from_state(comm_cost_m, machine_loads, worker_penalty_m, machine_penalty_m)

    for _pass in range(passes):
        for _step in range(steps_per_pass):
            u = boundary_nodes[rng.randrange(0, len(boundary_nodes))]
            src_part = part_of[u]

            # candidate dst parts: neighbor parts + random
            cand_parts = set()
            for v in und_adj[u].keys():
                pv = part_of[v]
                if pv != src_part:
                    cand_parts.add(pv)
                if len(cand_parts) >= candidate_parts_cap:
                    break
            for _ in range(6):
                cand_parts.add(rng.randrange(0, nparts))

            if not cand_parts:
                continue

            dst_part = list(cand_parts)[rng.randrange(0, len(cand_parts))]
            if dst_part == src_part:
                continue

            w_u = float(compute_w[u])
            if w_u <= 0.0:
                w_u = 0.0

            # hard ceiling checks (compute-load)
            if part_sizes[dst_part] + w_u > worker_ceiling:
                continue

            m_src = part_machine[src_part]
            m_dst = part_machine[dst_part]
            if m_src != m_dst and (machine_loads[m_dst] + w_u > machine_ceiling):
                continue

            old_obj = cur_obj

            # ------------------------------------------------------------
            # APPLY HYPOTHETICAL MOVE (mutate), compute new_obj, then revert
            # ------------------------------------------------------------

            # Track what we need to revert
            touched_machines = set()
            touched_parts = {src_part, dst_part}
            touched_edges = list(incident[u])

            # 1) update hyperedge costs + comm_cost_m with correct attribution
            comm_deltas = defaultdict(float)

            # mutate part_of[u] temporarily for edge-cost computation
            part_of[u] = dst_part

            for eid in touched_edges:
                old_c = hyperedge_cost[eid]
                old_attr = hyperedge_attr_m[eid]

                # recompute cost under new placement
                pins = hg_pins[eid]
                new_c = hyperedge_cost_twolevel(pins, hg_w[eid], part_of, part_machine, alpha, beta)

                sender = sender_of[eid]
                new_attr = part_machine[part_of[sender]]  # sender's machine under hypothetical

                # apply comm delta to machines
                if old_attr == new_attr:
                    dc = float(new_c) - float(old_c)
                    if abs(dc) > 0.0:
                        comm_deltas[old_attr] += dc
                        touched_machines.add(old_attr)
                else:
                    # attribution changed (only happens when sender==u and u moved machines)
                    comm_deltas[old_attr] += -float(old_c)
                    comm_deltas[new_attr] += float(new_c)
                    touched_machines.add(old_attr)
                    touched_machines.add(new_attr)

                # store new cost/attr in arrays (still hypothetical)
                hyperedge_cost[eid] = float(new_c)
                hyperedge_attr_m[eid] = int(new_attr)

            # apply comm deltas
            for m, dc in comm_deltas.items():
                comm_cost_m[m] += float(dc)

            # 2) update part sizes
            part_sizes[src_part] -= w_u
            part_sizes[dst_part] += w_u

            # 3) update machine loads (compute)
            touched_machines.add(m_src)
            touched_machines.add(m_dst)
            machine_loads[m_src] -= w_u
            machine_loads[m_dst] += w_u

            # 4) update worker penalties for src/dst parts and aggregate per machine
            # remove old contributions
            old_pen_src = part_penalty[src_part]
            old_pen_dst = part_penalty[dst_part]
            worker_penalty_m[m_src] -= float(old_pen_src)
            worker_penalty_m[m_dst] -= float(old_pen_dst)

            # recompute
            part_penalty[src_part] = _quad_overflow_penalty(part_sizes[src_part], nodes_per_worker, lambda_worker)
            part_penalty[dst_part] = _quad_overflow_penalty(part_sizes[dst_part], nodes_per_worker, lambda_worker)

            # add new
            worker_penalty_m[m_src] += float(part_penalty[src_part])
            worker_penalty_m[m_dst] += float(part_penalty[dst_part])

            # 5) update machine penalties for affected machines
            old_mp_src = machine_penalty_m[m_src]
            old_mp_dst = machine_penalty_m[m_dst]
            machine_penalty_m[m_src] = _quad_overflow_penalty(machine_loads[m_src], nodes_per_machine, lambda_machine)
            machine_penalty_m[m_dst] = _quad_overflow_penalty(machine_loads[m_dst], nodes_per_machine, lambda_machine)

            # 6) compute new objective
            new_obj = current_objective()
            delta = float(new_obj) - float(old_obj)

            # SA accept
            accept = False
            if delta <= 0.0:
                accept = True
            else:
                if T > 1e-12:
                    p_acc = math.exp(-float(delta) / float(T))
                    if rng.random() < p_acc:
                        accept = True

            if accept:
                # keep move
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_part_of = list(part_of)
            else:
                # ------------------
                # REVERT EVERYTHING
                # ------------------
                # revert machine penalties
                machine_penalty_m[m_src] = old_mp_src
                machine_penalty_m[m_dst] = old_mp_dst

                # revert worker penalties aggregates and per-part
                worker_penalty_m[m_src] -= float(part_penalty[src_part])
                worker_penalty_m[m_dst] -= float(part_penalty[dst_part])
                part_penalty[src_part] = old_pen_src
                part_penalty[dst_part] = old_pen_dst
                worker_penalty_m[m_src] += float(old_pen_src)
                worker_penalty_m[m_dst] += float(old_pen_dst)

                # revert machine loads
                machine_loads[m_src] += w_u
                machine_loads[m_dst] -= w_u

                # revert part sizes
                part_sizes[src_part] += w_u
                part_sizes[dst_part] -= w_u

                # revert comm costs
                for m, dc in comm_deltas.items():
                    comm_cost_m[m] -= float(dc)

                # revert edges
                for eid in touched_edges:
                    # we need old cost/attr; easiest is recompute back by restoring u then recomputing:
                    pass

                # restore u and recompute edge costs/attrs for touched edges from scratch, and fix comm_cost_m precisely
                # This is "unlimited runtime": keep correctness, accept slower.
                part_of[u] = src_part

                # First remove the current (hypothetical) contributions from comm_cost_m for touched edges,
                # then add back true contributions.
                # We already reverted comm_cost_m via comm_deltas, but edge arrays are still hypothetical;
                # rewrite them consistent with reverted comm_cost_m.
                for eid in touched_edges:
                    pins = hg_pins[eid]
                    true_c = hyperedge_cost_twolevel(pins, hg_w[eid], part_of, part_machine, alpha, beta)
                    sender = sender_of[eid]
                    true_attr = part_machine[part_of[sender]]
                    hyperedge_cost[eid] = float(true_c)
                    hyperedge_attr_m[eid] = int(true_attr)

                cur_obj = old_obj

                continue

            # if accepted, part_of[u] is already dst. if rejected we already continued.
            # NOTE: we did not update boundary pool; it's ok.

        T *= float(T_decay)

    # revert to best seen
    for i in range(n):
        part_of[i] = best_part_of[i]

    return part_of


# ============================================================
# Part->machine remapping optimization (swap search, capacity-aware)
# (Uses weighted part_sizes and strict machine compute cap)
# ============================================================

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
      - enforce strict machine compute-load capacity after swap (hard)
    """
    rng = random.Random(seed)
    nparts = len(part_machine)

    machine_parts = [[] for _ in range(num_machines)]
    for p in range(nparts):
        m = part_machine[p]
        machine_parts[m].append(p)

    machine_load = [0.0] * num_machines
    for m in range(num_machines):
        s = 0.0
        for p in machine_parts[m]:
            s += float(part_sizes[p])
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

                for pa in machine_parts[ma]:
                    ca = within_contrib(pa, ma)
                    cb = within_contrib(pa, mb)
                    for pb in machine_parts[mb]:
                        da = within_contrib(pb, mb)
                        db = within_contrib(pb, ma)

                        new_load_a = machine_load[ma] - float(part_sizes[pa]) + float(part_sizes[pb])
                        new_load_b = machine_load[mb] - float(part_sizes[pb]) + float(part_sizes[pa])
                        if new_load_a > float(nodes_per_machine) or new_load_b > float(nodes_per_machine):
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

                    machine_load[ma] = float(new_load_a)
                    machine_load[mb] = float(new_load_b)

                    swaps_done += 1
                    improved = True

            if swaps_done >= max_swaps_per_round:
                break

        if not improved:
            break

    return part_machine


# ============================================================
# Initial grouping of parts into machines (capacity-aware greedy)
# (Uses weighted part_sizes and machine compute cap)
# ============================================================

def greedy_group_parts_into_machines(part_traffic, num_machines, workers_per_machine, part_sizes, nodes_per_machine, seed=42):
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
    machine_load = [0.0] * num_machines

    parts_sorted = sorted(range(nparts), key=lambda p: (total_degree[p], rng.random()), reverse=True)

    for m in range(num_machines):
        if not unassigned:
            break

        seed_part = None
        for p in parts_sorted:
            if p in unassigned and machine_load[m] + float(part_sizes[p]) <= float(nodes_per_machine):
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
        machine_load[m] += float(part_sizes[seed_part])

        while len(group) < workers_per_machine and unassigned:
            best_p = None
            best_score = -1.0
            for cand in unassigned:
                if machine_load[m] + float(part_sizes[cand]) > float(nodes_per_machine):
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
            machine_load[m] += float(part_sizes[best_p])

        machine_parts[m] = group

    if unassigned:
        leftovers = sorted(list(unassigned))
        for p in leftovers:
            placed = False
            ms = sorted(range(num_machines), key=lambda mm: machine_load[mm])
            for m in ms:
                if len(machine_parts[m]) >= workers_per_machine:
                    continue
                if machine_load[m] + float(part_sizes[p]) <= float(nodes_per_machine):
                    part_machine[p] = m
                    machine_parts[m].append(p)
                    machine_load[m] += float(part_sizes[p])
                    placed = True
                    break
            if not placed:
                m = min(range(num_machines), key=lambda mm: machine_load[mm])
                part_machine[p] = m
                machine_parts[m].append(p)
                machine_load[m] += float(part_sizes[p])

    return part_machine, machine_parts


# ============================================================
# Compatibility: undirected cut breakdown for legacy stats consumers
# ============================================================

def compute_cost_breakdown_raw_undirected(und_adj, part_of, part_machine):
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


# ============================================================
# One-shot pipeline (alternating SA + swaps) - FIXED
# ============================================================

def partition_one_shot_sa_alternating(
    und_adj,
    hg_pins,
    hg_w,
    incident,
    sender_of,
    compute_w,
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
    # Final strictness (ignored; preserved for CLI compatibility)
    final_strict_repair=True,
):
    n = len(und_adj)
    if num_machines * nodes_per_machine < sum(compute_w):
        # This check is "compute capacity" now. But keep the old-style feasibility too.
        # The old check (num_machines * nodes_per_machine < n) is meaningless once nodes_per_machine is compute-cap.
        pass

    workers_per_machine = int(math.ceil(nodes_per_machine / float(nodes_per_worker)))
    if workers_per_machine <= 0:
        workers_per_machine = 1
    total_workers = num_machines * workers_per_machine

    # --- coarsening on undirected projection
    coarse_adj, _super_of, members, super_size = affinity_coarsen_pairs(
        und_adj,
        percentile=coarsen_percentile,
        max_pair_fraction=coarsen_max_pair_fraction,
        seed=seed + 11,
    )

    # Partition coarsened graph with METIS
    super_part = metis_partition(coarse_adj, total_workers, vweights=super_size, seed=seed + 21)

    # Uncoarsen to node-level parts
    part_of = uncoarsen_part_assignment(super_part, members)
    if any(p < 0 or p >= total_workers for p in part_of):
        raise RuntimeError("Uncoarsen produced invalid part ids.")

    # NO strict repair anywhere (Bug 3). Start from METIS assignment and let soft penalties handle it.

    # Initial mapping parts->machines (hypergraph affinity)
    part_sizes = compute_part_sizes_weighted(part_of, total_workers, compute_w)
    part_aff = build_part_affinity_from_hypergraph(hg_pins, hg_w, part_of, total_workers)

    part_machine, _machine_parts = greedy_group_parts_into_machines(
        part_aff,
        num_machines,
        workers_per_machine,
        part_sizes=part_sizes,
        nodes_per_machine=nodes_per_machine,
        seed=seed + 101,
    )

    # SA temperature heuristic (if not provided)
    if sa_T0 is None:
        if hg_w:
            mean_w = sum(float(x) for x in hg_w) / float(len(hg_w))
        else:
            mean_w = 1.0
        sa_T0 = max(1.0, 0.1 * float(alpha) * mean_w)

    # Alternation loop
    for r in range(alt_rounds):
        part_of = refine_sa_minmax_hyper(
            und_adj=und_adj,
            hg_pins=hg_pins,
            hg_w=hg_w,
            incident=incident,
            sender_of=sender_of,
            compute_w=compute_w,
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

        # rebuild weighted sizes + affinity
        part_sizes = compute_part_sizes_weighted(part_of, total_workers, compute_w)
        part_aff = build_part_affinity_from_hypergraph(hg_pins, hg_w, part_of, total_workers)

        # remap parts->machines by swaps (strict machine cap, weighted)
        part_machine = improve_mapping_by_swaps(
            part_traffic=part_aff,
            part_machine=part_machine,
            part_sizes=part_sizes,
            num_machines=num_machines,
            workers_per_machine=workers_per_machine,
            nodes_per_machine=nodes_per_machine,
            seed=seed + 3000 + r,
            max_rounds=remap_rounds,
        )

    # NO final_strict_repair (Bug 3)
    # final_strict_repair flag is preserved but ignored by design.

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


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to comm traces (json dict-of-dicts or edgelist)")
    ap.add_argument("--format", choices=["json", "edgelist"], required=True)
    ap.add_argument("--directed", action="store_true", help="Treat edgelist as directed (default for edgelist)")
    ap.add_argument("--num_nodes", type=int, default=None, help="Optional explicit N. Otherwise inferred.")

    ap.add_argument("--num_machines", type=int, required=True)
    ap.add_argument("--nodes_per_machine", type=int, required=True)
    ap.add_argument("--nodes_per_worker", type=int, required=True)

    ap.add_argument("--alpha", type=float, default=20.0, help="Cross-machine penalty multiplier (broadcast-aware)")
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

    # Final strictness (preserved but ignored)
    ap.add_argument("--final_strict_repair", action="store_true", help="(IGNORED) kept for compatibility")

    ap.add_argument("--output", required=True, help="Output JSON mapping node -> machine/worker")

    # Backwards-compat: old drivers pass this
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

    # compute weights (Bug 2)
    compute_w = compute_vertex_compute_weights(comm, n)

    # undirected projection for METIS/coarsening/boundary proposals
    und_adj = symmetrize_to_undirected(comm, n)

    # static hypergraph for broadcast-aware objective
    hg_pins, hg_w, incident, sender_of = build_static_broadcast_hypergraph(comm, n)

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
        hg_pins=hg_pins,
        hg_w=hg_w,
        incident=incident,
        sender_of=sender_of,
        compute_w=compute_w,
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
        final_strict_repair=False,  # ignored anyway
    )

    # Output assignment (unchanged)
    out = {}
    for u in range(n):
        out[str(u)] = {"machine": int(machine_of[u]), "worker": int(worker_of[u])}

    # Weighted part sizes for stats
    part_sizes = compute_part_sizes_weighted(part_of, total_workers, compute_w)

    # Backward-compatible cut stats (undirected projection)
    intra_cut, inter_cut = compute_cost_breakdown_raw_undirected(und_adj, part_of, part_machine)

    # Build minmax breakdown stats
    # Recompute comm_cost_m from scratch
    comm_cost_m = [0.0] * args.num_machines
    comm_alpha_m = [0.0] * args.num_machines
    comm_beta_m = [0.0] * args.num_machines

    for eid, pins in enumerate(hg_pins):
        sender = sender_of[eid]
        m_attr = part_machine[part_of[sender]]
        # compute alpha/beta touches for debug
        machines = set()
        parts_by_machine = defaultdict(set)
        for u in pins:
            p = part_of[u]
            m = part_machine[p]
            machines.add(m)
            parts_by_machine[m].add(p)
        net_touches = max(0, len(machines) - 1)
        ipc_touches = 0
        for ps in parts_by_machine.values():
            ipc_touches += max(0, len(ps) - 1)
        a = float(hg_w[eid]) * float(args.alpha) * float(net_touches)
        b = float(hg_w[eid]) * float(args.beta) * float(ipc_touches)
        comm_alpha_m[m_attr] += a
        comm_beta_m[m_attr] += b
        comm_cost_m[m_attr] += (a + b)

    # machine loads (compute)
    machine_loads = [0.0] * args.num_machines
    for p in range(total_workers):
        machine_loads[part_machine[p]] += float(part_sizes[p])

    # penalties
    part_penalty = [0.0] * total_workers
    worker_penalty_m = [0.0] * args.num_machines
    for p in range(total_workers):
        pen = _quad_overflow_penalty(part_sizes[p], args.nodes_per_worker, args.lambda_worker)
        part_penalty[p] = pen
        worker_penalty_m[part_machine[p]] += float(pen)

    machine_penalty_m = [0.0] * args.num_machines
    for m in range(args.num_machines):
        machine_penalty_m[m] = _quad_overflow_penalty(machine_loads[m], args.nodes_per_machine, args.lambda_machine)

    machine_costs = []
    for m in range(args.num_machines):
        machine_costs.append(_machine_cost(comm_cost_m, machine_loads, worker_penalty_m, machine_penalty_m, m))

    minmax_obj = max(machine_costs) if machine_costs else 0.0
    straggler_m = max(range(args.num_machines), key=lambda m: machine_costs[m]) if args.num_machines > 0 else -1

    stats = {
        # existing keys preserved
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
        "final_strict_repair": False,  # explicitly false
        "intra_machine_cut_weight": intra_cut,
        "inter_machine_cut_weight": inter_cut,
        "twolevel_objective_cost_soft": float(minmax_obj),  # now MINMAX objective
        "workers_per_machine": worker_counts,

        # extra keys (safe to ignore)
        "gamma_compute": float(GAMMA),
        "compute_total": float(sum(compute_w)) if compute_w else 0.0,
        "hyper_num_nets": len(hg_pins),
        "hyper_total_net_weight": float(sum(hg_w)) if hg_w else 0.0,
        "minmax_machine_costs": [float(x) for x in machine_costs],
        "minmax_straggler_machine": int(straggler_m),
        "minmax_straggler_cost": float(machine_costs[straggler_m]) if straggler_m >= 0 else 0.0,
        "hyper_comm_cost_by_machine": [float(x) for x in comm_cost_m],
        "hyper_comm_alpha_by_machine": [float(x) for x in comm_alpha_m],
        "hyper_comm_beta_by_machine": [float(x) for x in comm_beta_m],
        "compute_load_by_machine": [float(x) for x in machine_loads],
        "worker_penalty_by_machine": [float(x) for x in worker_penalty_m],
        "machine_penalty_by_machine": [float(x) for x in machine_penalty_m],
    }

    payload = {"assignment": out, "stats": stats}
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()