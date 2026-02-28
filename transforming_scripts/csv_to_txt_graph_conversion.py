import csv
from collections import defaultdict
from tqdm import tqdm

adj = defaultdict(set)

with open("twitch_gamers/large_twitch_edges.csv") as f:
    total_edges = sum(1 for _ in f) - 1  # minus header

with open("twitch_gamers/large_twitch_edges.csv", newline="") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for u, v in tqdm(reader, total=total_edges, desc="Reading edges"):
        u = int(u)
        v = int(v)
        adj[u].add(v)
        adj[v].add(u)  # undirected

with open("large_twitch_gamers_graph.txt", "w") as out:
    for u in tqdm(adj, desc="Writing vertices"):
        nbrs = list(adj[u])
        out.write(f"{u}\t{len(nbrs)}")
        for v in nbrs:
            out.write(f" {v}")
        out.write("\n")
