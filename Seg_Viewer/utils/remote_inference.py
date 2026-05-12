import streamlit as st
import paramiko
import time
import threading
from datetime import datetime


def render_gpu_training_ui():
    st.subheader("🖥️ Bitahub GPU 训练平台")
    
    with st.expander("🔗 SSH 连接配置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            ssh_host = st.text_input("SSH 地址", value=st.session_state.get('ssh_host', ''))
            ssh_port = st.number_input("SSH 端口", value=int(st.session_state.get('ssh_port', 22)), min_value=1, max_value=65535)
            ssh_user = st.text_input("用户名", value=st.session_state.get('ssh_user', ''))
        
        with col2:
            ssh_pass = st.text_input("密码", type="password", value=st.session_state.get('ssh_pass', ''))
            remote_work_dir = st.text_input("远程工作目录", 
                                          value=st.session_state.get('remote_work_dir', '~/Seg_Viewer'),
                                          help="远程服务器上的项目路径")
        
        st.session_state['ssh_host'] = ssh_host
        st.session_state['ssh_port'] = ssh_port
        st.session_state['ssh_user'] = ssh_user
        st.session_state['ssh_pass'] = ssh_pass
        st.session_state['remote_work_dir'] = remote_work_dir
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("🔌 测试连接", use_container_width=True):
            test_ssh_connection()
    
    with btn_col2:
        if st.button("💾 保存配置", use_container_width=True):
            save_ssh_config()
    
    with btn_col3:
        if st.button("📊 查看 GPU 状态", use_container_width=True):
            check_gpu_status()
    
    if 'ssh_connected' not in st.session_state:
        st.session_state['ssh_connected'] = False
    
    if st.session_state.get('ssh_connected'):
        st.success("✅ 已连接到远程服务器")
        
        st.divider()
        render_training_config_ui()


def test_ssh_connection():
    try:
        ssh = create_ssh_client()
        if ssh:
            stdin, stdout, stderr = ssh.exec_command('echo "Connection successful" && uname -a')
            output = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            
            if output and "successful" in output:
                st.success(f"✅ 连接成功！\n{output}")
                st.session_state['ssh_connected'] = True
                st.rerun()
            else:
                st.error(f"❌ 连接失败: {error}")
                st.session_state['ssh_connected'] = False
            ssh.close()
    except Exception as e:
        st.error(f"❌ 连接错误: {str(e)}")
        st.session_state['ssh_connected'] = False


def create_ssh_client():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=st.session_state.get('ssh_host', ''),
            port=int(st.session_state.get('ssh_port', 22)),
            username=st.session_state.get('ssh_user', ''),
            password=st.session_state.get('ssh_pass', ''),
            timeout=10
        )
        return ssh
    except Exception as e:
        st.error(f"创建SSH连接失败: {str(e)}")
        return None


def save_ssh_config():
    config = {
        'host': st.session_state.get('ssh_host', ''),
        'port': st.session_state.get('ssh_port', 22),
        'user': st.session_state.get('ssh_user', ''),
        'work_dir': st.session_state.get('remote_work_dir', '')
    }
    
    st.session_state['saved_ssh_config'] = config
    st.success("✅ 配置已保存到会话中")


def check_gpu_status():
    if not st.session_state.get('ssh_connected'):
        ssh = create_ssh_client()
        if not ssh:
            return
    else:
        ssh = create_ssh_client()
        if not ssh:
            return
    
    try:
        st.info("正在查询 GPU 状态...")
        
        gpu_cmd = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"
        stdin, stdout, stderr = ssh.exec_command(gpu_cmd)
        gpu_output = stdout.read().decode().strip()
        gpu_error = stderr.read().decode().strip()
        
        if gpu_error and "command not found" in gpu_error.lower():
            st.warning("⚠️ 远程服务器未安装 nvidia-smi 或无GPU")
            ssh.close()
            return
        
        if gpu_output:
            st.markdown("### 🎮 GPU 状态信息")
            gpu_lines = gpu_output.split('\n')
            for line in gpu_lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    with st.container():
                        cols = st.columns(6)
                        cols[0].metric("GPU ID", parts[0])
                        cols[1].metric("型号", parts[1][:20] + "..." if len(parts[1]) > 20 else parts[1])
                        cols[2].metric("显存使用", f"{parts[2]} / {parts[3]}")
                        cols[3].metric("GPU利用率", f"{parts[4]}%")
                        cols[4].metric("温度", f"{parts[5]}°C")
                        cols[5].markdown("")
                    st.divider()
        
        disk_cmd = "df -h / | tail -1 | awk '{print $4 \" 可用 / \" $2 \" 总计\"}'"
        stdin, stdout, stderr = ssh.exec_command(disk_cmd)
        disk_info = stdout.read().decode().strip()
        
        if disk_info:
            st.caption(f"💾 磁盘空间: {disk_info}")
        
        ssh.close()
    except Exception as e:
        st.error(f"查询GPU状态失败: {str(e)}")
        if ssh:
            ssh.close()


def render_training_config_ui():
    st.subheader("⚙️ 训练配置")
    
    with st.form("training_config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox(
                "选择模型",
                options=["VNet", "VNet_Decouple_Attention_ABC (SKCDF)", "VNet_DHC"],
                index=0,
                help="选择要训练的模型架构"
            )
            
            dataset_name = st.selectbox(
                "数据集",
                options=["synapse", "amos"],
                index=0,
                help="Synapse: 13类 | AMOS: 15类"
            )
            
            epochs = st.number_input("训练轮数 (Epochs)", value=100, min_value=1, max_value=1000)
            batch_size = st.number_input("Batch Size", value=2, min_value=1, max_value=16)
        
        with col2:
            learning_rate = st.number_input("学习率", value=0.01, format="%.4f", help="初始学习率")
            data_percentage = st.slider(
                "训练数据百分比",
                min_value=10,
                max_value=100,
                value=100,
                step=10,
                help="使用多少百分比的训练数据"
            )
            
            gpu_id = st.text_input("GPU ID", value="0", help="使用的GPU编号，多个用逗号分隔，如 '0,1'")
        
        submitted = st.form_submit_button("🚀 提交训练任务", use_container_width=True, type="primary")
        
        if submitted:
            submit_training_job(model_name, dataset_name, epochs, batch_size, 
                              learning_rate, data_percentage, gpu_id)


def submit_training_job(model_name, dataset_name, epochs, batch_size, lr, data_percent, gpu_id):
    ssh = create_ssh_client()
    if not ssh:
        return False
    
    try:
        work_dir = st.session_state.get('remote_work_dir', '~/Seg_Viewer').replace('~', '$HOME')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.split()[0].lower().replace('_', '')
        job_name = f"{safe_model_name}_{dataset_name}_{data_percent}p_{timestamp}"
        
        train_script = f"""#!/bin/bash
# Training Job: {job_name}
# Submitted at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

set -e

echo "=========================================="
echo "Starting Training Job: {job_name}"
echo "Model: {model_name}"
echo "Dataset: {dataset_name}"
echo "Epochs: {epochs}"
echo "Batch Size: {batch_size}"
echo "Learning Rate: {lr}"
echo "Data Percentage: {data_percent}%"
echo "GPU: CUDA_VISIBLE_DEVICES={gpu_id}"
echo "=========================================="

export CUDA_VISIBLE_DEVICES={gpu_id}
cd {work_dir}

echo "Current directory: $(pwd)"
echo "Python: $(which python)"
echo "GPU Info:"
nvidia-smi

echo ""
echo "Starting training..."
python train.py \\
    --model {model_name.split()[0] if '(' not in model_name else model_name.split('(')[1].split(')')[0]} \\
    --dataset {dataset_name} \\
    --epochs {epochs} \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --data_percent {data_percent} \\
    --save_dir ./ckpts/{dataset_name}/ \\
    --job_name {job_name}

echo ""
echo "Training completed!"
echo "Model saved to: ./ckpts/{dataset_name}/{job_name}.pth"

# Cleanup
echo "Job finished at: $(date)"
"""
        
        remote_script_path = f"/tmp/train_{job_name}.sh"
        
        sftp = ssh.open_sftp()
        with sftp.file(remote_script_path, 'w') as f:
            f.write(train_script)
        sftp.close()
        
        chmod_cmd = f"chmod +x {remote_script_path}"
        stdin, stdout, stderr = ssh.exec_command(chmod_cmd)
        stderr.read()
        
        nohup_cmd = f"nohup bash {remote_script_path} > ./logs/{job_name}.log 2>&1 & echo $!"
        stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {work_dir}/logs && {nohup_cmd}")
        pid = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        
        if pid and pid.isdigit():
            st.success(f"""
            ✅ **训练任务已成功提交！**
            
            - 📝 **任务名称**: `{job_name}`
            - 🔢 **进程ID**: `{pid}`
            - 🎯 **模型**: {model_name}
            - 📊 **数据集**: {dataset_name}
            - 🔄 **轮数**: {epochs} epochs
            - 💾 **日志文件**: `./logs/{job_name}.log`
            
            ---
            💡 **提示**:
            - 使用下方按钮查看训练进度和日志
            - 训练完成后模型将保存在 ckpts 目录
            """)
            
            if 'training_jobs' not in st.session_state:
                st.session_state['training_jobs'] = []
            
            st.session_state['training_jobs'].append({
                'name': job_name,
                'pid': pid,
                'model': model_name,
                'dataset': dataset_name,
                'status': 'running',
                'submitted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'log_file': f"./logs/{job_name}.log"
            })
        else:
            st.error(f"❌ 提交任务失败: {error}")
        
        ssh.close()
        return True
        
    except Exception as e:
        st.error(f"❌ 提交训练任务时出错: {str(e)}")
        if ssh:
            ssh.close()
        return False


def render_job_management_ui():
    st.subheader("📋 训练任务管理")
    
    jobs = st.session_state.get('training_jobs', [])
    
    if not jobs:
        st.info("暂无运行中的训练任务")
        return
    
    for i, job in enumerate(jobs):
        with st.expander(f"{'🟢' if job['status'] == 'running' else '🔴'} {job['name']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("状态", job['status'])
                st.metric("PID", job['pid'])
            
            with col2:
                st.metric("模型", job['model'][:30])
                st.metric("数据集", job['dataset'])
            
            with col3:
                st.metric("提交时间", job['submitted_at'])
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("查看日志", key=f"log_{i}", use_container_width=True):
                        view_training_log(job)
                
                with btn_col2:
                    if st.button("停止任务", key=f"stop_{i}", use_container_width=True, type="secondary"):
                        stop_training_job(i)


def view_training_log(job):
    ssh = create_ssh_client()
    if not ssh:
        return
    
    try:
        log_file = job.get('log_file', '')
        if log_file:
            cmd = f"tail -50 {log_file}"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            log_content = stdout.read().decode()
            error = stderr.read().decode()
            
            if log_content:
                st.code(log_content, language='bash')
            elif error:
                st.warning(f"无法读取日志: {error}")
            else:
                st.info("日志文件为空或还在生成中...")
        else:
            st.warning("未找到日志文件路径")
        
        ssh.close()
    except Exception as e:
        st.error(f"读取日志失败: {str(e)}")
        if ssh:
            ssh.close()


def stop_training_job(job_index):
    jobs = st.session_state.get('training_jobs', [])
    if job_index >= len(jobs):
        return
    
    job = jobs[job_index]
    ssh = create_ssh_client()
    if not ssh:
        return
    
    try:
        kill_cmd = f"kill {job['pid']} 2>/dev/null; ps -p {job['pid']} > /dev/null && echo 'running' || echo 'stopped'"
        stdin, stdout, stderr = ssh.exec_command(kill_cmd)
        status = stdout.read().decode().strip()
        
        if status == 'stopped':
            st.session_state['training_jobs'][job_index]['status'] = 'stopped'
            st.success(f"✅ 任务 {job['name']} 已停止")
            st.rerun()
        else:
            st.warning(f"⚠️ 任务可能仍在运行")
        
        ssh.close()
    except Exception as e:
        st.error(f"停止任务失败: {str(e)}")
        if ssh:
            ssh.close()


def render_gpu_inference_ui():
    st.subheader("🖥️ Bitahub GPU 平台")
    st.info("通过 SSH 连接到远程 GPU 服务器进行推理或训练")
    
    render_gpu_training_ui()
    
    if st.session_state.get('training_jobs'):
        st.divider()
        render_job_management_ui()
