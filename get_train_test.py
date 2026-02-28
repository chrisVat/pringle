# get_train_test.py

import argparse
import os
import random


def get_train_test(path, train_pct, seed):
    """
    Args:
        path (str): path to selected_nodes_64.txt
        train_pct (float): fraction in [0, 1], e.g. 0.8
        seed (int): RNG seed

    Returns:
        train_nodes (list[int])
        test_nodes (list[int])
    """
    with open(path, "r", encoding="utf-8") as f:
        nodes = [int(line.strip()) for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(nodes)

    split_idx = int(len(nodes) * train_pct)
    train_nodes = nodes[:split_idx]
    test_nodes = nodes[split_idx:]

    return train_nodes, test_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        default="node_selection/twitch_gamers/selected_nodes_64.txt",
        help="Path to selected_nodes_64.txt",
    )
    ap.add_argument(
        "--train_pct",
        type=float,
        default=0.2,
        help="Train fraction, e.g. 0.8",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    ap.add_argument(
        "--out_dir",
        default="train_test",
        help="Output directory",
    )
    args = ap.parse_args()

    train_nodes, test_nodes = get_train_test(
        path=args.path,
        train_pct=args.train_pct,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    base = os.path.basename(args.path)
    stem, _ = os.path.splitext(base)

    tag = f"train{args.train_pct:.2f}_seed{args.seed}"
    train_path = os.path.join(args.out_dir, f"{stem}_{tag}_train.txt")
    test_path = os.path.join(args.out_dir, f"{stem}_{tag}_test.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        for n in train_nodes:
            f.write(f"{n}\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for n in test_nodes:
            f.write(f"{n}\n")

    print("Train nodes:")
    print(train_nodes)
    print("\nTest nodes:")
    print(test_nodes)
    print(f"\nCounts: train={len(train_nodes)}, test={len(test_nodes)}")
    print(f"\nWrote:")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()