#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
import time

import paramiko


# --- CONFIG ---
USER = "ubuntu"
REGION = "us-east-2"
AWS_PROFILE = "pregel"

# Where the timing CSV lives on the master
REMOTE_QUERY_TIMES = "/tmp/query_times.csv"

# Where to save it locally
LOCAL_OUT_DIR = os.path.join(os.getcwd(), "query_times_pulls")

# Your PEM path
MASTER_KEY_PATH = r"C:\Users\chris\.ssh\pregel_master.pem"


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
        print("Error: 'aws' not found in PATH. Install/configure AWS CLI.", file=sys.stderr)
        return []


def get_master_dns():
    masters = get_aws_cli_instances("pregel_master")
    return masters[0] if masters else None


def sftp_pull_then_delete(hostname: str, remote_path: str, local_path: str):
    """
    Downloads remote_path to local_path, then deletes remote_path if download succeeds.
    Deletion is done over the same SSH connection.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=USER, key_filename=MASTER_KEY_PATH)

    try:
        # Pull
        sftp = ssh.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()

        # Delete remote after successful pull
        stdin, stdout, stderr = ssh.exec_command(f"rm -f {remote_path}")
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            err = stderr.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Remote delete failed (rc={rc}): {err}")

    finally:
        ssh.close()


def parse_query_times_csv(path: str):
    """
    Expected columns:
      source,query_seconds,supersteps,total_msgs,total_vadd
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            try:
                rows.append({
                    "source": int(r["source"]),
                    "query_seconds": float(r["query_seconds"]),
                    "supersteps": int(r.get("supersteps", "0") or 0),
                    "total_msgs": int(r.get("total_msgs", "0") or 0),
                    "total_vadd": int(r.get("total_vadd", "0") or 0),
                })
            except Exception:
                continue
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


def main():
    master = get_master_dns()
    if not master:
        print("Could not find a running master (tag=pregel_master).", file=sys.stderr)
        return 1

    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(LOCAL_OUT_DIR, f"query_times_{ts}.csv")

    print(f"Master: {master}")
    print(f"Pulling {REMOTE_QUERY_TIMES} -> {local_path}")
    print("Will delete remote file after successful pull.")

    try:
        sftp_pull_then_delete(master, REMOTE_QUERY_TIMES, local_path)
    except FileNotFoundError:
        print(f"Remote file not found: {REMOTE_QUERY_TIMES}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"SFTP pull/delete failed: {e}", file=sys.stderr)
        return 1

    rows = parse_query_times_csv(local_path)
    if not rows:
        print("Pulled file, but parsed 0 rows. Check CSV header/format.", file=sys.stderr)
        return 1

    times = sorted(r["query_seconds"] for r in rows)
    avg = sum(times) / len(times)

    p50 = percentile(times, 50)
    p90 = percentile(times, 90)
    p99 = percentile(times, 99)

    print()
    print(f"Queries: {len(times)}")
    print(f"Average query_seconds: {avg:.6f}")
    print(f"Median (p50): {p50:.6f}")
    print(f"p90: {p90:.6f}")
    print(f"p99: {p99:.6f}")
    print(f"Min/Max: {times[0]:.6f} / {times[-1]:.6f}")
    print()
    print(f"Saved: {local_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())