import paramiko
import time
import subprocess

# --- CONFIGURATION ---
USER = "ubuntu"
REGION = "us-east-2"

# Your specific PEM file paths
# MASTER_KEY_PATH = r"C:\\Users\\jason\\Documents\\Coursework\\214\\jasonpringle.pem"
# SLAVE_KEY_PATH = MASTER_KEY_PATH

MASTER_KEY_PATH = r"C:\Users\chris\.ssh\pregel_master.pem"
SLAVE_KEY_PATH  = r"C:\Users\chris\.ssh\pregel_slave.pem"




# hosts_text = hosts_text.replace("\n", "\\n")

# --- COMMANDS ---
# No more ENV_VARS needed here, we are relying on .bashrc!
MASTER_STOP_CMDS = f"""
"""

SLAVE_CLEAN_CMD = "ls"

MASTER_START_CMDS = """
"""

print(MASTER_STOP_CMDS)

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

# Returns (private, public)
def get_aws_cli_instance_ips(tag_value):
	"""Runs the AWS CLI to fetch Public DNS names for running instances matching a tag."""
	cmd = [
		"aws", "ec2", "describe-instances",
		"--region", REGION,
		"--profile", "pregel",  # Using your specific AWS profile
		"--filters", 
		f"Name=tag:Name,Values={tag_value}", 
		"Name=instance-state-name,Values=running",
		"--query", "Reservations[*].Instances[*].[PrivateIpAddress, PublicIpAddress]",
		"--output", "text"
	]
	
	try:
		result = subprocess.run(cmd, capture_output=True, text=True, check=True)
		arr = result.stdout.strip().split()
		return [(arr[i], arr[i+1]) for i in range(0, len(arr), 2)]
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

def get_active_instance_ips():
	print("Fetching active EC2 instance IP addresses via AWS CLI...")
	
	master_list = get_aws_cli_instance_ips("pregel_master")
	master_dns = master_list[0] if master_list else None

	slave_dns_list = get_aws_cli_instance_ips("pregel_slave*")

	return master_dns, slave_dns_list

def execute_ssh_command(hostname, commands, description, key_path):
	print(f"\n[{hostname}] --- {description} ---")
	
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	
	try:
		ssh.connect(hostname=hostname, username=USER, key_filename=key_path)
		
		# Take the multiline string and join the commands with semicolons
		# so they run sequentially on a single line.
		cmd_list = [commands.strip()]
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

def generateHostsText():
	print("Generating hosts file text...")
	master_ip, slave_ips = get_active_instance_ips()
	print(master_ip)
	print(slave_ips)
	numSlaves = len(slave_ips)

	names = []
	for i in range(numSlaves):
		if i>0:
			names.append(f"{slave_ips[i][0]} slave{i+1}")
		else:
			names.append(f"{slave_ips[i][0]} slave")
	s = "\n".join(names)


	# We just need private IPs...

	# Example stuff to write into /etc/hosts:
	return f"""
	# Jason waz here
	127.0.0.1 localhost
	{master_ip[0]} master
	{s}

	# The following lines are desirable for IPv6 capable hosts
	::1 ip6-localhost ip6-loopback
	fe00::0 ip6-localnet
	ff00::0 ip6-mcastprefix
	ff02::1 ip6-allnodes
	ff02::2 ip6-allrouters
	ff02::3 ip6-allhosts
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

	hosts_text = generateHostsText()
	write_hosts_text_cmd = f"sudo sh -c \"cat >/etc/hosts\" <<-EOF\n {hosts_text}"

	execute_ssh_command(master_node, write_hosts_text_cmd, "Writing master /etc/hosts", MASTER_KEY_PATH)

	for slave in slave_nodes:
		execute_ssh_command(slave, write_hosts_text_cmd, f"Writing {slave} /etc/hosts", SLAVE_KEY_PATH)

	# time.sleep(2)

	# execute_ssh_command(master_node, MASTER_START_CMDS, "Starting Hadoop & Running Pringle Script", MASTER_KEY_PATH)
	
	print("\nAutomation complete!")

if __name__ == "__main__":
	main()