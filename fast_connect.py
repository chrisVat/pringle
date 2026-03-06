"""
auto starts up master and slaves 
ur gonna want to get the .pem files and put them somewhere it can be loaded
i am running windows, may be slight differences across os
you'll need paramiko (pip install paramiko)
you'll also need aws cli
once you get aws cli run aws cli for the profile pregel
you'll need to generate a key to do so go to aws, iam, profile, (your name), security credentials, create access key, commandline
the gpts can help you with that for running this script. 

this will then do a full clean and start everything up. yay.
"""



import paramiko
import time
import subprocess

# --- CONFIGURATION ---
USER = "ubuntu"
REGION = "us-east-2"

# Your specific PEM file paths
MASTER_KEY_PATH = r"/Users/safiaboutaleb/Desktop/pregel_master.pem"
SLAVE_KEY_PATH = r"/Users/safiaboutaleb/Desktop/pregel_slave.pem"

# MASTER_KEY_PATH = r"C:\Users\chris\.ssh\pregel_master.pem"
# SLAVE_KEY_PATH = r"C:\Users\chris\.ssh\pregel_slave.pem"

# Set to True to pull a git branch and recompile on master + all slaves before starting.
SYNC_AND_RECOMPILE = True
GIT_BRANCH = "eval_comm_new"          # branch to checkout on all nodes
PRINGLE_DIR = "/home/ubuntu/pringle"
SSSP_DIR    = "/home/ubuntu/pringle/sssp"

# --- COMMANDS ---
# No more ENV_VARS needed here, we are relying on .bashrc!
MASTER_STOP_CMDS = """
stop-dfs.sh
stop-yarn.sh
rm -rf /usr/local/hadoop/data/hdfs/datanode/*
rm -rf /usr/local/hadoop/data/hdfs/namenode/*
hdfs namenode -format -force
"""

SLAVE_CLEAN_CMD = "rm -rf /usr/local/hadoop/data/hdfs/datanode/*"

MASTER_START_CMDS = """
start-dfs.sh
start-yarn.sh
hadoop fs -mkdir -p /largeTwitchFolder
hadoop fs -put large_twitch_gamers_graph.txt /largeTwitchFolder
"""

# cd /home/ubuntu/pringle/sssp/ && ./source_run.sh

def get_aws_cli_instances(tag_value):
    """Runs the AWS CLI to fetch Public DNS names for running instances matching a tag."""
    cmd = [
        "aws", "ec2", "describe-instances",
        "--region", REGION,
        "--profile", "pregel",  # Using your specific AWS profile
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
        print("Error: The 'aws' command was not found. Please ensure the AWS CLI is in your system's PATH.")
        return []

def get_active_instances():
    print("Fetching active EC2 instances via AWS CLI...")
    
    master_list = get_aws_cli_instances("pregel_master")
    master_dns = master_list[0] if master_list else None

    slave_dns_list = get_aws_cli_instances("pregel_slave*")

    return master_dns, slave_dns_list

def execute_ssh_command(hostname, commands, description, key_path):
    print(f"\n[{hostname}] --- {description} ---")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(hostname=hostname, username=USER, key_filename=key_path)
        
        # Take the multiline string and join the commands with semicolons
        # so they run sequentially on a single line.
        cmd_list = [c.strip() for c in commands.strip().split('\n') if c.strip()]
        chained_commands = " ; ".join(cmd_list)
        
        # Run everything in a single interactive bash execution (-c)
        full_command = f"bash -i -c 'source ~/.bashrc ; {chained_commands}'"
        
        # No more stdin.write() or stdin.close() needed!
        stdin, stdout, stderr = ssh.exec_command(full_command)
        
        # Stream standard output line-by-line as it happens!
        for line in stdout:
            print(f"  STDOUT: {line.strip()}")
            
        # Print any errors after
        for line in stderr:
            print(f"  STDERR: {line.strip()}")
            
        exit_status = stdout.channel.recv_exit_status()
            
        if exit_status == 0:
            print(f"[{hostname}] Success.")
        else:
            print(f"[{hostname}] Failed with status {exit_status}.")
            
    except Exception as e:
        print(f"[{hostname}] Connection or execution error: {e}")
    finally:
        ssh.close()


SYNC_CMDS = f"""
cd {PRINGLE_DIR} && git reset --hard HEAD~ && git fetch && git checkout {GIT_BRANCH} && git pull
cd {SSSP_DIR} && make clean && make
"""

def main():
    print("Starting Hadoop Cluster Reset Automation...")

    master_node, slave_nodes = get_active_instances()

    if not master_node:
        print("Error: Could not find a running master node. Exiting.")
        return
    if not slave_nodes:
        print("Warning: No running slave nodes found. Proceeding with master only.")

    print(f"Found Master: {master_node}")
    print(f"Found {len(slave_nodes)} Slaves: {', '.join(slave_nodes)}")

    if SYNC_AND_RECOMPILE:
        print(f"\n=== Syncing branch '{GIT_BRANCH}' and recompiling on all nodes ===")
        execute_ssh_command(master_node, SYNC_CMDS, f"git checkout {GIT_BRANCH} + make", MASTER_KEY_PATH)
        for slave in slave_nodes:
            execute_ssh_command(slave, SYNC_CMDS, f"git checkout {GIT_BRANCH} + make", SLAVE_KEY_PATH)

    execute_ssh_command(master_node, MASTER_STOP_CMDS, "Stopping Hadoop & Formatting Namenode", MASTER_KEY_PATH)

    for slave in slave_nodes:
        execute_ssh_command(slave, SLAVE_CLEAN_CMD, "Cleaning Slave Datanode Directory", SLAVE_KEY_PATH)

    time.sleep(2)

    execute_ssh_command(master_node, MASTER_START_CMDS, "Starting Hadoop & Running Pringle Script", MASTER_KEY_PATH)
    
    print("\nAutomation complete!")

if __name__ == "__main__":
    main()