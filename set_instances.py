import boto3
import time
import socket
import paramiko
from botocore.exceptions import ClientError
import subprocess

# =========================
# CONFIG
# For Master's ~/hosts file, need to manually add entries for all slaves and master itself
# =========================
AWS_PROFILE = "pregel"
REGION = "us-east-2"

AMI_ID = "ami-058c3349bade00a81"
INSTANCE_TYPE = "t2.xlarge"
KEY_NAME = "jasonpringle"   # EC2 key pair name in AWS
PEM_PATH = "/Users/0225n/Documents/UCLA/26Winter/214_Pringle/jasonpringle.pem"

SECURITY_GROUP_IDS = ["sg-06fd397a9fb37f016"]
SUBNET_ID = "subnet-0ece47ab7f2e26908"

INSTANCE_NAME_TAG = "pregel_slave_"
HOSTNAME = "slave"

SSH_USER = "ubuntu"
IP_HOSTNAME_TXT_FILE = "instance_ip_hostname.txt"
MASTER_PUBLIC_IP = "3.135.195.197"

NUMBER_OF_INSTANCES_ALREADY = 3 # number of slaves u have rn
NUMBER_OF_INSTANCES_TO_LAUNCH = 4 # total number of slaves
NUMBER_OF_WORKER_PER_INSTANCE = 4 # number of workers to run on each slave (based on number of cores, leave some room for OS)


def build_user_data(hostname: str) -> str:
    """
    Optional: set hostname during boot using cloud-init.
    Even though we will also set it again over SSH, this helps make it consistent.
    """
    return f"""#cloud-config
preserve_hostname: true
hostname: {hostname}
manage_etc_hosts: false
"""


def wait_for_instance_running(ec2_client, instance_id: str):
    print(f"[INFO] Waiting for instance {instance_id} to enter 'running' state...")
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    print(f"[INFO] Instance {instance_id} is now running.")


def get_instance_info(ec2_client, instance_id: str):
    desc = ec2_client.describe_instances(InstanceIds=[instance_id])
    inst = desc["Reservations"][0]["Instances"][0]

    return {
        "instance_id": inst["InstanceId"],
        "private_ip": inst.get("PrivateIpAddress"),
        "public_ip": inst.get("PublicIpAddress"),
        "state": inst["State"]["Name"],
    }


def wait_for_ssh(host: str, port: int = 22, timeout: int = 300):
    """
    Wait until port 22 is reachable.
    """
    print(f"[INFO] Waiting for SSH on {host}:{port} ...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"[INFO] SSH is ready on {host}:{port}")
                return True
        except OSError:
            time.sleep(5)

    return False


def execute_ssh_command(hostname: str, username: str, key_path: str, command: str, description: str = ""):
    print(f"\n[{hostname}] --- {description or 'Running command'} ---")
    print(f"[CMD] {command}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=hostname,
            username=username,
            key_filename=key_path,
            timeout=20,
        )

        stdin, stdout, stderr = ssh.exec_command(command)

        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        exit_status = stdout.channel.recv_exit_status()

        if out:
            print("[STDOUT]")
            print(out)

        if err:
            print("[STDERR]")
            print(err)

        if exit_status == 0:
            print(f"[{hostname}] Success.")
            return True
        else:
            print(f"[{hostname}] Failed with exit status {exit_status}.")
            return False

    except Exception as e:
        print(f"[{hostname}] SSH error: {e}")
        return False
    finally:
        ssh.close()

def set_key_via_ssh(public_ip: str, hostname: str):
    write_key_cmd = f"""ssh-keygen -f '/home/ubuntu/.ssh/known_hosts' -R 'localhost'"""
    execute_ssh_command(
        hostname=public_ip,
        username=SSH_USER,
        key_path=PEM_PATH,
        command=write_key_cmd,
        description=f"Adding pubkey to {hostname} authorized_keys"
    )   
    # write_key_cmd = f"""
    #     mkdir -p ~/.ssh &&
    #     chmod 700 ~/.ssh &&

    #     if [ ! -f ~/.ssh/id_rsa ]; then
    #         ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""
    #     fi &&

    #     cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys &&
    #     echo '{PUBKEY}' >> ~/.ssh/authorized_keys &&
    #     chmod 600 ~/.ssh/authorized_keys
    #     """.strip()

    # 


def set_hostname_via_ssh(public_ip: str, hostname: str):
    """
    Set hostname on the remote machine via SSH.
    This updates current hostname and /etc/hosts entries.
    """
    cmd = f"""
        sudo sh -c "echo {hostname} > /etc/hostname" && \
        sudo hostnamectl set-hostname {hostname} && \
        hostname
        """.strip()

    return execute_ssh_command(
        hostname=public_ip,
        username=SSH_USER,
        key_path=PEM_PATH,
        command=cmd,
        description=f"Setting hostname to {hostname}"
    )
def set_hosts_file_via_ssh(public_ip: str):
    with open("./hosts.txt", "w") as f:
        f.write(f"master slots={NUMBER_OF_WORKER_PER_INSTANCE}\n")
        for i in range(1, NUMBER_OF_INSTANCES_TO_LAUNCH + 1):
            if(i==1):
                f.write(f"{HOSTNAME} slots={NUMBER_OF_WORKER_PER_INSTANCE}\n")
            else:
                f.write(f"{HOSTNAME}{i} slots={NUMBER_OF_WORKER_PER_INSTANCE}\n")
    with open("./slaves.txt", "w") as f:
        f.write(f"master\n")
        for i in range(1, NUMBER_OF_INSTANCES_TO_LAUNCH + 1):
            if(i==1):
                f.write(f"{HOSTNAME}\n")
            else:
                f.write(f"{HOSTNAME}{i}\n")
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-i", PEM_PATH,
        "./hosts.txt",
        f"{SSH_USER}@{public_ip}:~/hosts"
    ]

    subprocess.run(scp_cmd, check=True)
    scp2_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-i", PEM_PATH,
        "./slaves.txt",
        f"{SSH_USER}@{public_ip}:/usr/local/hadoop/etc/hadoop/slaves"
    ]

    subprocess.run(scp2_cmd, check=True)


def set_etc_hosts_file_via_ssh(public_ip: str, local_hosts_file: str):
    """
    Upload a local hosts txt file to the remote machine, then insert its
    content into /etc/hosts right before the IPv6 section without overwriting
    the original file.
    """

    print(f"\n[{public_ip}] --- Uploading {local_hosts_file} and inserting into /etc/hosts ---")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=public_ip,
            username=SSH_USER,
            key_filename=PEM_PATH,
            timeout=20,
        )

        remote_hosts_file = "/tmp/cluster_hosts.txt"

        # 1. upload local txt to remote
        sftp = ssh.open_sftp()
        sftp.put(local_hosts_file, remote_hosts_file)
        sftp.close()

        # 2. insert uploaded file before IPv6 section
        cmd = r"""
sudo python3 - <<'PY'
from pathlib import Path

hosts_path = Path("/etc/hosts")
custom_path = Path("/tmp/cluster_hosts.txt")

original = hosts_path.read_text()
custom = custom_path.read_text().rstrip() + "\n\n"

marker = "# The following lines are desirable for IPv6 capable hosts"

if marker in original:
    new_content = original.replace(marker, custom + marker, 1)
else:
    new_content = original.rstrip() + "\n\n" + custom

Path("/tmp/hosts.new").write_text(new_content)
PY

sudo mv /tmp/hosts.new /etc/hosts
cat /etc/hosts
""".strip()

        stdin, stdout, stderr = ssh.exec_command(cmd)

        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        exit_status = stdout.channel.recv_exit_status()

        if out:
            print("[STDOUT]")
            print(out)

        if err:
            print("[STDERR]")
            print(err)
        
        if exit_status == 0:
            print(f"[{public_ip}] /etc/hosts updated successfully.")
        else:
            print(f"[{public_ip}] Failed with exit status {exit_status}.")
            return False
        
        for i in range(NUMBER_OF_INSTANCES_TO_LAUNCH):
            hostname = ""
            if(i == 0):
                hostname = HOSTNAME  
            else:
                hostname = HOSTNAME + str(i + 1)
            
            write_hosts_cmd = f'''
scp -o StrictHostKeyChecking=no /etc/hosts {SSH_USER}@{hostname}:/tmp/hosts && \
ssh -o StrictHostKeyChecking=no {SSH_USER}@{hostname} "sudo mv /tmp/hosts /etc/hosts && cat /etc/hosts"
'''.strip()
            
            execute_ssh_command(
            hostname=public_ip,
            username=SSH_USER,
            key_path=PEM_PATH,
            command=write_hosts_cmd,
            description=f"Setting hostname to {hostname}"
        )

    
    except Exception as e:
        print(f"[{public_ip}] SSH/SFTP error: {e}")
        return False

    finally:
        ssh.close()

def append_ip_hostname_record_txt(privatepath: str, instance_info: dict, hostname: str):
    line = f"{instance_info['private_ip']} {hostname}\n"
    with open(privatepath, "a") as f:
        f.write(line)
    print(f"[INFO] {line.strip()}")


def launch_instance():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
    ec2 = session.client("ec2")
    with open(IP_HOSTNAME_TXT_FILE, "w"):
        pass
    for i in range(NUMBER_OF_INSTANCES_ALREADY+1, NUMBER_OF_INSTANCES_TO_LAUNCH+1):
        hostname = HOSTNAME + str(i)
        instance_name_tag = INSTANCE_NAME_TAG + str(i)
        user_data = build_user_data(hostname)

        try:
            response = ec2.run_instances(
                ImageId=AMI_ID,
                InstanceType=INSTANCE_TYPE,
                KeyName=KEY_NAME,
                MinCount=1,
                MaxCount=1,
                SecurityGroupIds=SECURITY_GROUP_IDS,
                SubnetId=SUBNET_ID,
                UserData=user_data,
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": instance_name_tag},
                            {"Key": "Hostname", "Value": hostname},
                        ],
                    }
                ],
            )
        except ClientError as e:
            print("[ERROR] Failed to launch instance:")
            print(e)
            return None

        instance_id = response["Instances"][0]["InstanceId"]
        print(f"[INFO] Launched instance: {instance_id}")

        wait_for_instance_running(ec2, instance_id)
        info = get_instance_info(ec2, instance_id)

        print("[INFO] Instance details:")
        print(f"  Instance ID : {info['instance_id']}")
        print(f"  Private IP  : {info['private_ip']}")
        print(f"  Public IP   : {info['public_ip']}")

        if not info["public_ip"]:
            print("[ERROR] Instance has no public IP, cannot SSH from your local machine.")
            return info

        if not wait_for_ssh(info["public_ip"], timeout=300):
            print("[ERROR] SSH did not become available in time.")
            return info
        
        set_key_via_ssh(info["public_ip"], hostname)
        ok = set_hostname_via_ssh(info["public_ip"], hostname)
        if not ok:
            print("[WARNING] SSH hostname setup failed, but instance was created.")

        append_ip_hostname_record_txt(IP_HOSTNAME_TXT_FILE, info, hostname)
    
    set_etc_hosts_file_via_ssh(MASTER_PUBLIC_IP, IP_HOSTNAME_TXT_FILE)
    set_hosts_file_via_ssh(MASTER_PUBLIC_IP)
    return info


if __name__ == "__main__":
    result = launch_instance()
    
    if result:
        print("\n[DONE]")
        print(result)