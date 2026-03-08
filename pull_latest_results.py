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

# Where to save it locally
LOCAL_OUT_DIR = os.path.join(os.getcwd(), "query_times_pulls")

# Your PEM path
# MASTER_KEY_PATH = r"C:\Users\chris\.ssh\pregel_master.pem"
# MASTER_KEY_PATH = "/Users/safiaboutaleb/Desktop/pregel_master.pem"
MASTER_KEY_PATH = r"C:\Users\safia\OneDrive\Desktop\pregel_master.pem"

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


def find_remote_query_files(hostname: str) -> list[str]:
    """Returns list of /tmp/query_times_*.csv paths on the remote master."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=USER, key_filename=MASTER_KEY_PATH)
    try:
        _, stdout, _ = ssh.exec_command("ls /tmp/query_times_*.csv 2>/dev/null")
        out = stdout.read().decode().strip()
        return out.split("\n") if out else []
    finally:
        ssh.close()


def sftp_pull_then_delete(hostname: str, remote_path: str, local_path: str):
    """Downloads remote_path to local_path, then deletes the remote file."""
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


def parse_query_times_csv(path: str):
    """
    Expected columns:
      source,query_seconds,supersteps,total_msgs,total_vadd
    Stops at the first blank or comment line (stats block).
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r or not r.get("source", "").strip():
                break
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


def append_stats(local_path: str, rows: list):
    """Appends a stats block to the bottom of the saved CSV."""
    times_by_source = [(r["source"], r["query_seconds"]) for r in rows]
    times_by_source.sort(key=lambda x: x[0])
    times = sorted(r["query_seconds"] for r in rows)

    avg = sum(times) / len(times)
    median = percentile(times, 50)
    p99 = percentile(times, 99)

    source_nodes = ", ".join(str(s) for s, _ in times_by_source)
    individual_times = ", ".join(f"{t:.6f}" for _, t in times_by_source)

    with open(local_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"average: {avg:.6f}\n")
        f.write(f"median: {median:.6f}\n")
        f.write(f"p99: {p99:.6f}\n")
        f.write("\n")
        f.write(f"source nodes: {source_nodes}\n")
        f.write(f"individual times: {individual_times}\n")


def main():
    master = get_master_dns()
    if not master:
        print("Could not find a running master (tag=pregel_master).", file=sys.stderr)
        return 1

    print(f"Master: {master}")

    remote_files = find_remote_query_files(master)
    if not remote_files:
        print("No /tmp/query_times_*.csv files found on remote.", file=sys.stderr)
        return 1

    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    for remote_path in remote_files:
        # e.g. /tmp/query_times_theboogalo3__1_1_1000.csv -> query_times_theboogalo3__1_1_1000_20260305_120000.csv
        remote_name = os.path.basename(remote_path)          # query_times_<label>.csv
        stem = remote_name[len("query_times_"):-len(".csv")] # <label>
        local_name = f"query_times_{ts}_{stem}.csv"
        local_path = os.path.join(LOCAL_OUT_DIR, local_name)

        print(f"Pulling {remote_path} -> {local_path}")
        try:
            sftp_pull_then_delete(master, remote_path, local_path)
        except FileNotFoundError:
            print(f"Remote file not found: {remote_path}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"SFTP pull/delete failed: {e}", file=sys.stderr)
            continue

        rows = parse_query_times_csv(local_path)
        if not rows:
            print("  Warning: parsed 0 rows. Check CSV header/format.", file=sys.stderr)
            continue

        append_stats(local_path, rows)

        times = sorted(r["query_seconds"] for r in rows)
        print(f"  Queries : {len(times)}")
        print(f"  average : {sum(times)/len(times):.6f}")
        print(f"  median  : {percentile(times, 50):.6f}")
        print(f"  p99     : {percentile(times, 99):.6f}")
        print(f"  Saved   : {local_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
