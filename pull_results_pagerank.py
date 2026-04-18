#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
import time

import paramiko


USER = "ubuntu"
REGION = "us-east-2"
AWS_PROFILE = "pregel"

LOCAL_OUT_DIR = os.path.join(os.getcwd(), "pagerank_time_pulls")
MASTER_KEY_PATH = "/Users/safiaboutaleb/Desktop/pregel_master.pem"


def get_aws_cli_instances(tag_value: str):
    cmd = [
        "aws", "ec2", "describe-instances",
        "--region", REGION,
        "--profile", AWS_PROFILE,
        "--filters",
        f"Name=tag:Name,Values={tag_value}",
        "Name=instance-state-name,Values=running",
        "--query", "Reservations[*].Instances[*].PublicDnsName",
        "--output", "text",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = result.stdout.strip()
        return out.split() if out else []
    except subprocess.CalledProcessError as e:
        print(f"AWS CLI error: {e.stderr.strip()}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("Error: aws CLI not found.", file=sys.stderr)
        return []


def get_master_dns():
    masters = get_aws_cli_instances("pregel_master")
    return masters[0] if masters else None


def sftp_pull_then_delete(hostname: str, remote_path: str, local_path: str):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=USER, key_filename=MASTER_KEY_PATH)

    try:
        sftp = ssh.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()

        _, stdout, stderr = ssh.exec_command(f"rm -f {remote_path}")
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            err = stderr.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Remote delete failed (rc={rc}): {err}")
    finally:
        ssh.close()


def parse_pagerank_times_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r or not r.get("query_seconds", "").strip():
                continue
            rows.append({
                "query_seconds": float(r["query_seconds"]),
                "supersteps": int(r.get("supersteps", "0") or 0),
                "total_msgs": int(r.get("total_msgs", "0") or 0),
                "total_vadd": int(r.get("total_vadd", "0") or 0),
            })
    return rows


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def append_stats(local_path: str, rows: list):
    times = sorted(r["query_seconds"] for r in rows)

    avg = sum(times) / len(times)
    median = percentile(times, 50)
    p99 = percentile(times, 99)

    with open(local_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"average: {avg:.6f}\n")
        f.write(f"median: {median:.6f}\n")
        f.write(f"p99: {p99:.6f}\n")
        f.write("\n")
        f.write("individual times: " + ", ".join(f"{t:.6f}" for t in times) + "\n")

def find_latest_remote_query_file(hostname: str) -> str | None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=USER, key_filename=MASTER_KEY_PATH)
    try:
        cmd = r"ls -t /tmp/query_times_*.csv 2>/dev/null | head -n 1"
        _, stdout, _ = ssh.exec_command(cmd)
        out = stdout.read().decode().strip()
        return out if out else None
    finally:
        ssh.close()

def main():
    master = get_master_dns()
    if not master:
        print("Could not find a running master (tag=pregel_master).", file=sys.stderr)
        return 1

    remote_path = find_latest_remote_query_file(master)
    if not remote_path:
        print("No /tmp/query_times_*.csv file found on remote.", file=sys.stderr)
        return 1

    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    remote_name = os.path.basename(remote_path)
    local_path = os.path.join(LOCAL_OUT_DIR, f"{ts}_{remote_name}")

    print(f"Master: {master}")
    print(f"Pulling {remote_path} -> {local_path}")

    try:
        sftp_pull_then_delete(master, remote_path, local_path)
    except Exception as e:
        print(f"Failed to pull remote file: {e}", file=sys.stderr)
        return 1

    rows = parse_pagerank_times_csv(local_path)
    if not rows:
        print("Parsed 0 rows. Check the remote file format.", file=sys.stderr)
        return 1

    append_stats(local_path, rows)

    times = sorted(r["query_seconds"] for r in rows)
    print(f"Runs    : {len(times)}")
    print(f"average : {sum(times)/len(times):.6f}")
    print(f"median  : {percentile(times, 50):.6f}")
    print(f"p99     : {percentile(times, 99):.6f}")
    print(f"Saved   : {local_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())