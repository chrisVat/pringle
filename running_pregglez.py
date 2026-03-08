"""
running_pregglez.py - Run all pregglenator variants on src_9783.edgelist

Edit the CONFIG section below to set your topology params, then run:
    python running_pregglez.py
"""

import subprocess
import sys
import os

# ──────────────────────────────────────────────────────────────
# CONFIG - edit these
# ──────────────────────────────────────────────────────────────
INPUT         = "pregglenator_ready/src_9783.edgelist"
FORMAT        = "edgelist"
DIRECTED      = True   # pass --directed flag

NUM_MACHINES        = 4
NUM_WORKERS         = 4    # workers per machine (used by all variants)

SEED = 42

OUTPUT_DIR = "pregglenator_ready/outputs_8"

# Toggle which variants to run
RUN_PREGGLENATOR             = True
RUN_PREGGLENATOR_LOOSE       = False
RUN_PREGGLENATOR_ONESHOT     = False
RUN_PREGGLENATOR_GAMER_MODE  = False
RUN_PREGGLENATOR_GAMER_STRICT= False
RUN_METIS                    = True
# ──────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_DIR = "the_pregglenator_62000_v15"

PYTHON = sys.executable

def base_args(output_name):
    args = [
        "--input",            INPUT,
        "--format",           FORMAT,
        "--num_machines",      str(NUM_MACHINES),
        "--seed",              str(SEED),
        "--output",           os.path.join(OUTPUT_DIR, output_name),
    ]
    if DIRECTED:
        args.append("--directed")
    return args


VARIANTS = [v for v in [
    RUN_PREGGLENATOR and {
        "name": "pregglenator",
        "script": f"{BASE_DIR}/pregglenator.py",
        "extra_args": ["--num_workers", str(NUM_WORKERS)],
    },
    RUN_PREGGLENATOR_LOOSE and {
        "name": "pregglenator_loose",
        "script": f"{BASE_DIR}/pregglenator_loose.py",
        "extra_args": [
            "--num_workers",         str(NUM_WORKERS),
            "--workers_per_machine", str(NUM_WORKERS),
            "--worker_size_tol",     "0.10",
        ],
    },
    RUN_PREGGLENATOR_ONESHOT and {
        "name": "pregglenator_oneshot",
        "script": f"{BASE_DIR}/pregglenator_oneshot.py",
        "extra_args": [
            "--num_workers",            str(NUM_WORKERS),
            "--alpha",                  "20.0",
            "--beta",                   "1.0",
            "--alt_rounds",             "10",
            "--remap_rounds",           "12",
            "--sa_passes",              "10",
            "--sa_T_decay",             "0.85",
            "--slack_factor",           "1.05",
            "--lambda_worker",          "10.0",
            "--lambda_machine",         "10.0",
            "--coarsen_percentile",     "95.0",
            "--coarsen_max_pair_fraction", "0.45",
            # "--final_strict_repair",  # uncomment to enable strict repair at end
        ],
    },
    RUN_PREGGLENATOR_GAMER_MODE and {
        "name": "pregglenator_gamer_mode",
        "script": f"{BASE_DIR}/pregglenator_gamer_mode.py",
        "extra_args": [
            "--num_workers",          str(NUM_WORKERS),
            "--refine_machine_iters", "200",
            "--refine_worker_iters",  "100",
        ],
    },
    RUN_PREGGLENATOR_GAMER_STRICT and {
        "name": "pregglenator_gamer_mode_strict",
        "script": f"{BASE_DIR}/pregglenator_gamer_mode_strict.py",
        "extra_args": [
            "--workers_per_machine",  str(NUM_WORKERS),  # REQUIRED for strict
            "--refine_machine_iters", "200",
            "--refine_worker_iters",  "100",
        ],
    },
    RUN_METIS and {
        "name": "metis",
        "script": f"{BASE_DIR}/metis.py",
        "extra_args": ["--num_workers", str(NUM_WORKERS)],
    },
] if v]


def run_variant(variant):
    cmd = [PYTHON, variant["script"]] + base_args(f"{variant['name']}.json") + variant["extra_args"]
    print(f"\n{'='*60}")
    print(f"Running: {variant['name']}")
    print(f"CMD: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARNING] {variant['name']} exited with code {result.returncode}")
    else:
        print(f"[OK] {variant['name']} done")


if __name__ == "__main__":
    for v in VARIANTS:
        run_variant(v)
    print("\nAll variants complete. Outputs in:", OUTPUT_DIR)
