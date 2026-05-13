import paramiko
import os

def deploy_to_bitahub(ssh_host, ssh_port, ssh_user, ssh_pass, remote_dir="~/Seg_Viewer"):
    
    print(f"🔗 连接 {ssh_host}:{ssh_port} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=ssh_host, port=ssh_port, username=ssh_user, password=ssh_pass)
    
    sftp = ssh.open_sftp()
    
    local_base = r"d:\PythonProject2\Seg_Viewer"
    
    files_to_upload = [
        ("models/SKCDF.py", "models/SKCDF.py"),
        ("models/vnet.py", "models/vnet.py"),
        ("models/DHC/vnet_flat.py", "models/DHC/vnet_flat.py"),
        ("models/DHC/vnet_dst.py", "models/DHC/vnet_dst.py"),
        ("models/DHC/unet.py", "models/DHC/unet.py"),
        ("utils/__init__.py", "utils/__init__.py"),
        ("utils/config.py", "utils/config.py"),
        ("utils/evaluation_metrics.py", "utils/evaluation_metrics.py"),
    ]
    
    remote_dir_expanded = remote_dir.replace("~", "/root")
    
    for local_rel, remote_rel in files_to_upload:
        local_path = os.path.join(local_base, local_rel)
        remote_path = f"{remote_dir_expanded}/{remote_rel}"
        
        remote_folder = os.path.dirname(remote_path)
        
        try:
            sftp.mkdir(remote_folder)
        except:
            pass
        
        if os.path.exists(local_path):
            print(f"📤 上传: {local_rel} → {remote_rel}")
            sftp.put(local_path, remote_path)
        else:
            print(f"⚠️ 跳过(不存在): {local_rel}")
    
    sftp.close()
    
    stdin, stdout, stderr = ssh.exec_command(f"ls -la {remote_dir_expanded}/ && ls -la {remote_dir_expanded}/models/ && ls -la {remote_dir_expanded}/utils/")
    print("\n✅ 云端文件列表:")
    print(stdout.read().decode())
    
    ssh.close()
    print("\n🎉 部署完成！")


if __name__ == "__main__":
    deploy_to_bitahub(
        ssh_host="xj-member.bitahub.com",
        ssh_port=42141,
        ssh_user="root",
        ssh_pass="你的密码",
        remote_dir="~/Seg_Viewer"
    )
