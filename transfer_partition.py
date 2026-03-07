"""
Converts a partition JSON to txt and SCPs it to all master/slave nodes.

Usage:
    python transfer_partition.py                          # uses default JSON
    python transfer_partition.py -f path/to/file.json
    python transfer_partition.py -f file.json -n my_partition.txt
"""

import argparse
import os
import subprocess
import paramiko
from transforming_scripts.global_ranking_json_mine import convert_ranking

# --- CONFIGURATION ---
USER = "ubuntu"
REGION = "us-east-2"

MASTER_KEY_PATH = r"C:\Users\chris\.ssh\pregel_master.pem"
SLAVE_KEY_PATH  = r"C:\Users\chris\.ssh\pregel_slave.pem"

DEFAULT_JSON     = "the_pregglenator_62000_v15/pregglenator_compute_only.json"
PARTITIONS_DIR   = "partitions_txt"
REMOTE_DIR       = "/home/ubuntu/pringle/sssp/"
REMOTE_NAME = DEFAULT_JSON.split("/")[-1].replace(".json", ".txt") 
#REMOTE_NAME      = "large_twitch_graph_partition.txt"


# --- AWS ---

def get_aws_cli_instances(tag_value):
    cmd = [
        "aws", "ec2", "describe-instances",
        "--region", REGION,
        "--profile", "pregel",
        "--filters",
        f"Name=tag:Name,Values={tag_value}",
        "Name=instance-state-name,Values=running",
        "--query", "Reservations[*].Instances[*].PublicDnsName",
        "--output", "text"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split()
    except subprocess.CalledProcessError as e:
        print(f"AWS CLI Error fetching {tag_value}: {e.stderr}")
        return []
    except FileNotFoundError:
        print("Error: 'aws' command not found. Make sure AWS CLI is in your PATH.")
        return []

def get_active_instances():
    print("Fetching active EC2 instances via AWS CLI...")
    master_list    = get_aws_cli_instances("pregel_master")
    master_dns     = master_list[0] if master_list else None
    slave_dns_list = get_aws_cli_instances("pregel_slave*")
    return master_dns, slave_dns_list


# --- TRANSFER ---

def scp_file(hostname, key_path, local_path, remote_name):
    remote_path = REMOTE_DIR + remote_name
    print(f"\n[{hostname}] Uploading {local_path} -> {remote_path}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname=hostname, username=USER, key_filename=key_path)
        sftp = ssh.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        print(f"[{hostname}] Done.")
    except Exception as e:
        print(f"[{hostname}] Error: {e}")
    finally:
        ssh.close()


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Convert partition JSON to txt and SCP to cluster.")
    parser.add_argument("-f", "--file", default=DEFAULT_JSON, help=f"Path to input JSON (default: {DEFAULT_JSON})")
    parser.add_argument("-n", "--name", default=REMOTE_NAME, help=f"Remote filename on server (default: {REMOTE_NAME})")
    args = parser.parse_args()

    json_path = args.file
    txt_name  = os.path.splitext(os.path.basename(json_path))[0] + ".txt"
    txt_path  = os.path.join(PARTITIONS_DIR, txt_name)

    os.makedirs(PARTITIONS_DIR, exist_ok=True)
    if os.path.exists(txt_path):
        print(f"Found existing {txt_path}, skipping conversion.")
    else:
        print(f"Converting {json_path} -> {txt_path}")
        convert_ranking(json_path, txt_path)

    master_node, slave_nodes = get_active_instances()
    if not master_node:
        print("Error: no running master node found.")
        return

    print(f"Master : {master_node}")
    print(f"Slaves : {', '.join(slave_nodes) if slave_nodes else 'none'}")

    scp_file(master_node, MASTER_KEY_PATH, txt_path, args.name)
    for slave in slave_nodes:
        scp_file(slave, SLAVE_KEY_PATH, txt_path, args.name)

    print("\nTransfer complete!")

if __name__ == "__main__":
    main()
