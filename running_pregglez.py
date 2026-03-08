"""
running_pregglez.py - Run all pregglenator variants on all src_*.edgelist files
"""

import subprocess
import sys
import os
import glob

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
INPUT_DIR = "pregglenator_ready"
FORMAT = "edgelist"
DIRECTED = True

NUM_MACHINES = 4
NODES_PER_MACHINE = 42030
NODES_PER_WORKER = 10510 
WORKERS_PER_MACHINE = 4

SEED = 42

OUTPUT_DIR = "pregglenator_ready/outputs"
# ──────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = "the_pregglenator_62000_v15"
PYTHON = sys.executable


def base_args(input_file, output_name):
    args = [
        "--input", input_file,
        "--format", FORMAT,
        "--num_machines", str(NUM_MACHINES),
        "--nodes_per_machine", str(NODES_PER_MACHINE),
        "--nodes_per_worker", str(NODES_PER_WORKER),
        "--seed", str(SEED),
        "--output", os.path.join(OUTPUT_DIR, output_name),
    ]
    if DIRECTED:
        args.append("--directed")
    return args


VARIANTS = [
    {
        "name": "pregglenator",
        "script": f"{BASE_DIR}/pregglenator.py",
        "extra_args": [],
    },
    {
        "name": "pregglenator_loose",
        "script": f"{BASE_DIR}/pregglenator_loose.py",
        "extra_args": [
            "--workers_per_machine", str(WORKERS_PER_MACHINE),
            "--worker_size_tol", "0.10",
        ],
    },
    {
        "name": "pregglenator_oneshot",
        "script": f"{BASE_DIR}/pregglenator_oneshot.py",
        "extra_args": [
            "--alpha", "20.0",
            "--beta", "1.0",
            "--alt_rounds", "10",
            "--remap_rounds", "12",
            "--sa_passes", "10",
            "--sa_T_decay", "0.85",
            "--slack_factor", "1.05",
            "--lambda_worker", "10.0",
            "--lambda_machine", "10.0",
            "--coarsen_percentile", "95.0",
            "--coarsen_max_pair_fraction", "0.45",
        ],
    },
    {
        "name": "pregglenator_gamer_mode",
        "script": f"{BASE_DIR}/pregglenator_gamer_mode.py",
        "extra_args": [
            "--refine_machine_iters", "200",
            "--refine_worker_iters", "100",
        ],
    },
    {
        "name": "pregglenator_gamer_mode_strict",
        "script": f"{BASE_DIR}/pregglenator_gamer_mode_strict.py",
        "extra_args": [
            "--workers_per_machine", str(WORKERS_PER_MACHINE),
            "--refine_machine_iters", "200",
            "--refine_worker_iters", "100",
        ],
    },
]


def run_variant(input_file, variant):
    src_name = os.path.basename(input_file).replace(".edgelist", "")
    output_name = f"{src_name}_{variant['name']}.json"

    cmd = [PYTHON, variant["script"]] + base_args(input_file, output_name) + variant["extra_args"]

    print(f"\n{'='*60}")
    print(f"Running {variant['name']} on {src_name}")
    print(f"CMD: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[WARNING] {variant['name']} failed on {src_name}")
    else:
        print(f"[OK] {variant['name']} finished for {src_name}")


if __name__ == "__main__":

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "src_*.edgelist")))

    print(f"Found {len(input_files)} edgelist files")

    for input_file in input_files:
        for variant in VARIANTS:
            run_variant(input_file, variant)

    print("\nAll variants complete. Outputs in:", OUTPUT_DIR)