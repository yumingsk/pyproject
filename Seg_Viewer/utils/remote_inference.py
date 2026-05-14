import streamlit as st
import paramiko
import socket
import os
import tempfile
import hashlib
import nibabel as nib
import numpy as np
from datetime import datetime

#remote_inference.py
def create_ssh_client():
    ssh_host = st.session_state.get('ssh_host', '')
    ssh_port = int(st.session_state.get('ssh_port', 22))
    ssh_user = st.session_state.get('ssh_user', '')
    ssh_pass = st.session_state.get('ssh_pass', '')
    
    if not all([ssh_host, ssh_user, ssh_pass]):
        st.error("❌ 请先填写完整的 SSH 连接信息")
        return None
    
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        socket.setdefaulttimeout(15)
        ssh.connect(ssh_host, port=ssh_port, username=ssh_user, password=ssh_pass)
        
        return ssh
        
    except socket.gaierror:
        st.error(f"❌ DNS 解析失败: 无法解析主机名 '{ssh_host}'")
    except paramiko.AuthenticationException:
        st.error(f"❌ 认证失败: 用户名或密码错误")
    except paramiko.SSHException as e:
        st.error(f"❌ SSH 连接失败: {str(e)}")
    except socket.timeout:
        st.error(f"❌ 连接超时: 无法连接到 {ssh_host}:{ssh_port}")
    except Exception as e:
        st.error(f"❌ 连接错误: {str(e)}")
    
    return None


def test_ssh_connection():
    with st.spinner("🔄 正在测试 SSH 连接..."):
        ssh = create_ssh_client()
        
        if ssh:
            try:
                stdin, stdout, stderr = ssh.exec_command("uname -a && hostname && nproc", timeout=10)
                result = stdout.read().decode().strip()
                
                if result:
                    st.success("✅ SSH 连接成功！")
                    st.code(result, language="bash")
                    st.session_state['ssh_connected'] = True
                    
                    work_dir = st.session_state.get('remote_work_dir', '/workspace')
                    
                    check_cmd = f'test -d {work_dir} && echo "EXISTS" || echo "NOT_EXISTS"'
                    stdin, stdout, stderr = ssh.exec_command(check_cmd)
                    exists_result = stdout.read().decode().strip()
                    
                    if exists_result == 'EXISTS':
                        st.info(f"✅ 远程工作目录存在: {work_dir}")
                    else:
                        st.warning(f"⚠️ 远程工作目录不存在: {work_dir}，请检查路径设置")
                else:
                    st.error("⚠️ 连接成功但无法获取服务器信息")
                
            except Exception as e:
                st.error(f"❌ 测试命令执行失败: {e}")
            
            finally:
                ssh.close()
        else:
            st.session_state['ssh_connected'] = False


def check_gpu_status():
    ssh = create_ssh_client()
    if not ssh:
        return
    
    try:
        with st.spinner("🔄 正在查询 GPU 状态..."):
            gpu_cmd = "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
            stdin, stdout, stderr = ssh.exec_command(gpu_cmd, timeout=10)
            result = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            
            if result:
                lines = result.strip().split('\n')
                gpu_data = []
                
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpu_data.append({
                                'id': parts[0],
                                'name': parts[1],
                                'temp': parts[2],
                                'gpu_util': parts[3],
                                'mem_used': parts[4],
                                'mem_total': parts[5]
                            })
                
                if gpu_data:
                    cols = st.columns(len(gpu_data))
                    
                    for idx, gpu in enumerate(gpu_data):
                        with cols[idx]:
                            st.metric(
                                label=f"GPU {gpu['id']}: {gpu['name'][:20]}",
                                value=f"{gpu['gpu_util']}%",
                                delta=f"🌡️ {gpu['temp']}°C | 💾 {gpu['mem_used']}/{gpu['mem_total']} MB"
                            )
                else:
                    st.warning("⚠️ 未检测到 GPU 信息")
            else:
                st.warning("⚠️ 无法获取 GPU 状态 (可能未安装 nvidia-smi 或无 GPU)")
                if err:
                    st.code(err[:200], language="bash")
            
            ssh.close()
            
    except Exception as e:
        st.error(f"❌ GPU 查询失败: {e}")


def scan_server_models():
    ssh = create_ssh_client()
    if not ssh:
        return
    
    try:
        work_dir = st.session_state.get('remote_work_dir', '/workspace')
        
        with st.spinner("🔄 正在扫描服务器模型..."):
            scan_cmd = f'find {work_dir}/ckpts -name "*.pth" -type f 2>/dev/null | sort'
            stdin, stdout, stderr = ssh.exec_command(scan_cmd, timeout=30)
            result = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            
            if result:
                model_files = [f.strip() for f in result.split('\n') if f.strip()]
                
                if model_files:
                    model_names = []
                    model_paths = {}
                    
                    for mfile in model_files:
                        name = os.path.basename(mfile)
                        model_names.append(name)
                        model_paths[name] = mfile
                    
                    st.success(f"✅ 找到 {len(model_names)} 个模型:")
                    for name in model_names:
                        size_cmd = f'ls -lh "{model_paths[name]}" | awk \'{{print $5}}\''
                        stdin, stdout, stderr = ssh.exec_command(size_cmd)
                        size_result = stdout.read().decode().strip()
                        
                        if size_result:
                            st.text(f"  📦 {name} ({size_result})")
                        else:
                            st.text(f"  📦 {name}")
                    
                    st.session_state['server_models'] = model_names
                    st.session_state['server_model_paths'] = model_paths
                else:
                    st.warning(f"⚠️ 目录 {work_dir}/ckpts/ 中没有找到 .pth 模型文件")
                    st.info("💡 提示: 请确保模型文件放在服务器的 {work_dir}/ckpts/ 目录下")
            else:
                st.warning(f"⚠️ 未找到模型目录或目录为空: {work_dir}/ckpts/")
                if err and "No such file" in err:
                    st.info(f"💡 提示: 请先在服务器上创建 {work_dir}/ckpts/ 目录并放入 .pth 模型文件")
            
            ssh.close()
            
    except Exception as e:
        st.error(f"❌ 模型扫描失败: {e}")


def smart_upload_to_server(ssh, uploaded_file, dataset_type, file_type="File"):
    remote_base = f"/workspace/data/{dataset_type}"
    filename = uploaded_file.name
    
    safe_filename = filename.replace(' ', '_').replace('-', '_')
    remote_path = f"{remote_base}/{safe_filename}"
    
    mkdir_cmd = f'mkdir -p {remote_base}'
    ssh.exec_command(mkdir_cmd)
    
    sftp = ssh.open_sftp()
    try:
        try:
            remote_stat = sftp.stat(remote_path)
            remote_size = remote_stat.st_size
            
            local_bytes = uploaded_file.getvalue()
            local_size = len(local_bytes)
            
            if local_size == remote_size:
                sftp.close()
                st.info(f"✅ [{file_type}] 文件已存在且大小相同，跳过上传: {safe_filename}")
                return remote_path, False, True
                
            else:
                st.warning(f"⚠️ [{file_type}] 服务器已有同名文件但大小不同:\n"
                          f"  本地: {local_size / 1024 / 1024:.1f}MB\n"
                          f"  远程: {remote_size / 1024 / 1024:.1f}MB\n"
                          f"  将覆盖远程文件")
        except FileNotFoundError:
            pass
        
        progress_bar = st.progress(0, text=f"📤 正在上传 [{file_type}]...")
        
        def callback(transferred, total):
            if total > 0:
                progress = transferred / total
                progress_bar.progress(min(progress, 0.99), 
                                    text=f"📤 上传中 [{file_type}]... {transferred / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        sftp.put(tmp_path, remote_path, callback=callback)
        
        os.unlink(tmp_path)
        progress_bar.progress(1.0, text=f"✅ [{file_type}] 上传完成!")
        
        local_size_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        st.success(f"✅ [{file_type}] 上传成功: {safe_filename} ({local_size_mb:.1f} MB)")
        
        sftp.close()
        return remote_path, True, False
        
    except Exception as e:
        sftp.close()
        st.error(f"❌ [{file_type}] 上传失败: {e}")
        raise e


def render_gpu_inference_ui():
    st.subheader("🖥️ Bitahub 远程 GPU 推理")
    st.info("""
    **工作流程**: 
    📤 上传CT → 🖥️ 服务器GPU推理 → 📥 下载结果 → 🎨 本地显示
    
    文件将保存到云端 `data/{数据集类型}/` 目录，自动检测重复文件跳过上传
    """)
    
    with st.expander("🔗 SSH 连接配置", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            ssh_host = st.text_input("SSH 地址", value=st.session_state.get('ssh_host', 'xj-member.bitahub.com'))
            ssh_port = st.number_input("SSH 端口", value=int(st.session_state.get('ssh_port', 42141)), min_value=1, max_value=65535)
            ssh_user = st.text_input("用户名", value=st.session_state.get('ssh_user', 'root'))
        
        with col2:
            ssh_pass = st.text_input("密码", type="password", value=st.session_state.get('ssh_pass', ''))
            remote_work_dir = st.text_input("远程工作目录", 
                                          value=st.session_state.get('remote_work_dir', '/workspace'),
                                          help="远程服务器上的项目路径 (模型在 {此目录}/ckpts/ 下)")
        
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
        if st.button("📊 GPU 状态", use_container_width=True):
            check_gpu_status()
    
    with btn_col3:
        if st.button("📂 扫描模型", use_container_width=True):
            scan_server_models()
    
    if 'ssh_connected' not in st.session_state:
        st.session_state['ssh_connected'] = False
    
    if st.session_state.get('ssh_connected'):
        st.success("✅ 已连接到远程服务器")
    
    st.divider()
    
    server_models = st.session_state.get('server_models', [])
    
    if not server_models:
        st.warning("⚠️ 请先点击「扫描模型」获取服务器上的可用模型")
        selected_models = []
    else:
        st.markdown("**选择预测模型 (服务器端)**")
        selected_models = st.multiselect(
            "服务器模型列表",
            options=server_models,
            default=[server_models[0]] if server_models else None,
            help="从服务器上已有的 .pth 模型中选择"
        )
    
    st.divider()
    
    dataset_type = st.radio(
        "📁 数据集类型",
        options=["synapse", "amos"],
        index=0,
        horizontal=True,
        help="决定文件上传到服务器的 data/synapse/ 或 data/amos/ 目录"
    )
    st.session_state['dataset_type'] = dataset_type
    
    raw_file = st.file_uploader(
        f"1️⃣ 上传 Raw CT 图像 (.nii.gz) → 将存入 data/{dataset_type}/", 
        type=['nii', 'nii.gz'], 
        key="remote_raw"
    )
    gt_file = st.file_uploader(
        f"2️⃣ 上传 Ground Truth [可选] (.nii.gz) → 将存入 data/{dataset_type}/", 
        type=['nii', 'nii.gz'], 
        key="remote_gt"
    )
    
    alpha = st.slider("分割层透明度", 0.0, 1.0, 0.5)

    if raw_file:
        size_mb = len(raw_file.getvalue()) / 1024 / 1024
        st.info(f"📁 已选择: **{raw_file.name}** ({size_mb:.1f} MB) → `data/{dataset_type}/`")

        prev_file = st.session_state.get('remote_raw_name')
        if prev_file != raw_file.name:
            st.session_state['remote_raw_name'] = raw_file.name
            st.session_state['pre_upload_done'] = False

        if not st.session_state.get('pre_upload_done', False):
            ssh_pre = create_ssh_client()
            if ssh_pre:
                with st.spinner(f"⏳ 预上传中 ({size_mb:.1f}MB)..."):
                    try:
                        r_path, _, skipped = smart_upload_to_server(
                            ssh_pre, raw_file, dataset_type, "预上传"
                        )
                        st.session_state['pre_upload_remote_path'] = r_path
                        st.session_state['pre_upload_done'] = True
                    except Exception as e:
                        st.warning(f"预上传失败，推理时重试: {e}")
                    finally:
                        ssh_pre.close()
    else:
        st.info("请上传 CT 图像文件 (.nii.gz)")
        st.session_state['pre_upload_done'] = False

    run_enabled = raw_file is not None and selected_models
    run_text = "🚀 开始远程GPU预测" if run_enabled else "⏳ 请先上传图像并选择模型"
    if st.button(run_text, type="primary", use_container_width=True,
                disabled=(not raw_file or not selected_models)):
        run_remote_inference(raw_file, gt_file, selected_models, alpha, dataset_type)
    
    inference_log = st.session_state.get('inference_log', [])
    if inference_log:
        st.divider()
        st.subheader("📋 推理日志")
        
        log_text = '\n'.join(inference_log)
        has_error = any('❌' in line for line in inference_log)
        
        if has_error:
            st.error("⚠️ 上次运行存在错误，请查看下方日志")
        
        st.code(log_text, language='log')
        
        col_clear, _ = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ 清除日志", use_container_width=True):
                st.session_state.pop('inference_log', None)
                st.rerun()
    
    remote_raw_file = st.session_state.get('remote_raw_file')
    if remote_raw_file is not None:
        st.divider()
        st.subheader("切片导航")
        
        try:
            import nibabel as nib
            import numpy as np
            
            raw_data = nib.load(remote_raw_file).get_fdata(dtype=np.float32)
            dims = raw_data.shape
            
            if dims[0] > 1 and dims[1] > 1 and dims[2] > 1:
                idx_w = st.slider("矢状面 (Sagittal)", 0, dims[0] - 1, dims[0] // 2, key="gpu_sagittal")
                idx_h = st.slider("冠状面 (Coronal)", 0, dims[1] - 1, dims[1] // 2, key="gpu_coronal")
                idx_d = st.slider("横断面 (Axial)", 0, dims[2] - 1, dims[2] // 2, key="gpu_axial")
                
                st.session_state['gpu_slice_idx'] = (idx_w, idx_h, idx_d)
            else:
                st.warning("图像尺寸异常，无法启用切片导航")
        except Exception as e:
            st.error(f"加载切片导航失败: {e}")
    else:
        st.info("📌 请先上传并推理 CT 图像以启用切片导航")


def save_uploaded_file(uploaded_file, suffix):
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}")
    with open(tmp_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    return tmp_path


def run_remote_inference(raw_file, gt_file, selected_models, alpha, dataset_type):
    status = st.status("🔄 初始化远程推理...", expanded=True)

    log_messages = []

    def log(msg, level='info'):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f'[{timestamp}] {msg}'
        log_messages.append(log_msg)
        if level == 'error':
            st.error(msg)
        elif level == 'success':
            st.success(msg)
        elif level == 'warning':
            st.warning(msg)
        else:
            st.info(msg)

    ssh = create_ssh_client()
    if not ssh:
        return

    try:
        status.update(label=f"📤 步骤1/4: 上传 CT 图像到 data/{dataset_type}/ ...")

        if st.session_state.get('pre_upload_done'):
            remote_raw_path = st.session_state.get('pre_upload_remote_path', '')
            log(f'⚡ 使用预上传文件，跳过传输', 'success')
            raw_skipped = True
        else:
            remote_raw_path, raw_uploaded, raw_skipped = smart_upload_to_server(
                ssh, raw_file, dataset_type, "Raw CT"
            )

        remote_gt_path = None
        if gt_file:
            status.update(label=f"📤 步骤1.5/4: 上传 GT 到 data/{dataset_type}/ ...")
            remote_gt_path, _, _ = smart_upload_to_server(
                ssh, gt_file, dataset_type, "Ground Truth"
            )

        work_dir = st.session_state.get('remote_work_dir', '/workspace')

        status.update(label="📦 步骤2/4: 检查环境...")

        # ⭐ 修改点 1：检查我们刚写好的 infer_single.py 和 evaluate_Ntimes.py
        check_scripts_cmd = f'test -f {work_dir}/infer_single.py && test -f {work_dir}/evaluate_Ntimes.py && echo "READY" || echo "MISSING"'
        stdin, stdout, stderr = ssh.exec_command(check_scripts_cmd)
        scripts_ready = stdout.read().decode().strip()

        if scripts_ready != 'READY':
            log(f'❌ 云端缺少脚本: {work_dir}/infer_single.py 或 evaluate_Ntimes.py\n请先确保这两个文件在云端根目录',
                'error')
            st.session_state['inference_log'] = log_messages
            ssh.close()
            return

        log(f'✅ 环境检查通过', 'success')

        model_paths = st.session_state.get('server_model_paths', {})
        st.session_state['pred_results'] = {}
        st.session_state['cloud_eval_logs'] = {}
        st.session_state['remote_pred_paths'] = {}
        progress_bar = st.progress(0)

        input_basename = os.path.basename(remote_raw_path)
        name, ext = os.path.splitext(input_basename)
        if ext == '.gz':
            name, _ = os.path.splitext(name)

        local_cache_base = os.path.join(os.getcwd(), 'result')
        os.makedirs(local_cache_base, exist_ok=True)

        for i, model_name in enumerate(selected_models):
            model_path = model_paths.get(model_name, '')
            safe_name = model_name.replace('.', '_').replace('-', '_')
            pred_filename = f'pred_{name}.nii.gz'
            eval_filename = f'eval_{name}.txt'

            local_model_dir = os.path.join(local_cache_base, safe_name)
            local_pred_path = os.path.join(local_model_dir, pred_filename)
            local_eval_path = os.path.join(local_model_dir, eval_filename)

            output_dir = f'/workspace/predictions/{safe_name}'
            output_nii = f'{output_dir}/{pred_filename}'
            remote_eval_log = f'{output_dir}/eval_result_{name}.txt'

            ssh.exec_command(f'mkdir -p {output_dir}')

            # ======== 情况0：本地已有预测结果 + 评估结果 ========
            if os.path.exists(local_pred_path) and os.path.exists(local_eval_path):
                try:
                    pred_data = nib.load(local_pred_path).get_fdata(dtype=np.float32).astype(np.int16)
                    st.session_state['pred_results'][model_name] = pred_data
                    with open(local_eval_path, 'r') as f:
                        st.session_state['cloud_eval_logs'][model_name] = f.read()
                    log(f'⚡ [{model_name}] 本地缓存命中，秒加载: {local_pred_path}', 'success')
                    progress_bar.progress((i + 1) / len(selected_models))
                    continue
                except Exception as e:
                    log(f'⚠️ [{model_name}] 本地缓存损坏，重新获取: {e}', 'warning')

            check_cmd = f'test -f {output_nii} && echo "EXISTS" || echo "NOT_EXISTS"'
            stdin, stdout, stderr = ssh.exec_command(check_cmd)
            exists_result = stdout.read().decode().strip()

            # ======== 情况1：云端已存在 ========
            if exists_result == 'EXISTS':
                log(f'☁️ [{model_name}] 云端结果已存在: {output_nii}', 'info')

                os.makedirs(local_model_dir, exist_ok=True)
                sftp = ssh.open_sftp()
                try:
                    sftp.get(output_nii, local_pred_path)
                    pred_data = nib.load(local_pred_path).get_fdata(dtype=np.float32).astype(np.int16)
                    st.session_state['pred_results'][model_name] = pred_data
                    st.session_state['remote_pred_paths'][model_name] = output_nii
                    log(f'📥 [{model_name}] 预测结果已缓存到本地: {local_pred_path}', 'success')

                    if gt_file and remote_gt_path:
                        eval_exists_cmd = f'test -f {remote_eval_log} && echo "EVAL_EXISTS" || echo "NO_EVAL"'
                        stdin, stdout, stderr = ssh.exec_command(eval_exists_cmd)
                        eval_check = stdout.read().decode().strip()

                        if eval_check == 'EVAL_EXISTS' and os.path.exists(local_eval_path):
                            with open(local_eval_path, 'r') as f:
                                cached_eval = f.read()
                            st.session_state['cloud_eval_logs'][model_name] = cached_eval
                            log(f'⚡ [{model_name}] 评估结果已有，跳过运行', 'success')
                        elif eval_check == 'EVAL_EXISTS':
                            status.update(label=f'📊 下载评估 [{model_name}]...')
                            sftp.get(remote_eval_log, local_eval_path)
                            with open(local_eval_path, 'r') as f:
                                eval_result = f.read()
                            st.session_state['cloud_eval_logs'][model_name] = eval_result
                            log(f'📥 [{model_name}] 评估结果已下载并缓存', 'success')
                        else:
                            status.update(label=f'📊 运行评估 [{model_name}]...')
                            eval_cmd = (
                                f'source /opt/conda/etc/profile.d/conda.sh && conda activate base && '
                                f'cd {work_dir} && '
                                f'python evaluate_Ntimes.py '
                                f'--task {dataset_type} '
                                f'--pred {output_nii} '
                                f'--label {remote_gt_path} | tee {remote_eval_log}'
                            )
                            stdin, stdout, stderr = ssh.exec_command(eval_cmd, timeout=120)
                            eval_result = stdout.read().decode()
                            st.session_state['cloud_eval_logs'][model_name] = eval_result
                            with open(local_eval_path, 'w') as f:
                                f.write(eval_result)
                            log(f'📥 [{model_name}] 评估完成并缓存到本地', 'success')

                    progress_bar.progress((i + 1) / len(selected_models))
                    sftp.close()
                    continue

                except Exception as e:
                    log(f'⚠️ [{model_name}] 云端下载失败: {e}，将重新推理', 'warning')
                    sftp.close()

            # ======== 情景 B：云端没有结果，需要跑模型 ========
            log(f'🖥️ 步骤3/4: 运行云端推理 [{model_name}]... ({i + 1}/{len(selected_models)})', 'info')

            # ⭐ 修改点 2：使用 && 串联 infer_single.py 和 evaluate_Ntimes.py
            if gt_file and remote_gt_path:
                status.update(label=f'🖥️ 运行连招：推理 + 评估 [{model_name}]...')
                full_cmd = (
                    f'source /opt/conda/etc/profile.d/conda.sh && conda activate base && '
                    f'export CUDA_VISIBLE_DEVICES=0 && '
                    f'cd {work_dir} && '
                    f'python infer_single.py --task {dataset_type} --input {remote_raw_path} --output {output_nii} --model {model_path} '
                    f'&& '
                    f'python evaluate_Ntimes.py --task {dataset_type} --pred {output_nii} --label {remote_gt_path} | tee {remote_eval_log}'
                )
            else:
                status.update(label=f'🖥️ 运行纯推理 (无GT) [{model_name}]...')
                full_cmd = (
                    f'source /opt/conda/etc/profile.d/conda.sh && conda activate base && '
                    f'export CUDA_VISIBLE_DEVICES=0 && '
                    f'cd {work_dir} && '
                    f'python infer_single.py --task {dataset_type} --input {remote_raw_path} --output {output_nii} --model {model_path}'
                )

            # 这里超时时间稍微设长一点，防止大型 CT 推理太久
            stdin, stdout, stderr = ssh.exec_command(full_cmd, timeout=800)

            # 实时读取云端日志
            cmd_output = ""
            for line in stdout:
                cmd_output += line

            err = stderr.read().decode().strip()
            if err:
                log(f'[WARN] {err[:300]}', 'warning')

            if gt_file and remote_gt_path:
                eval_exists_cmd = f'test -f {remote_eval_log} && echo "EVAL_EXISTS" || echo "NO_EVAL"'
                stdin, stdout, stderr = ssh.exec_command(eval_exists_cmd)
                eval_check = stdout.read().decode().strip()

                if eval_check == 'EVAL_EXISTS':
                    sftp_read = ssh.open_sftp()
                    try:
                        sftp_read.get(remote_eval_log, local_eval_path)
                        with open(local_eval_path, 'r') as f:
                            eval_result = f.read()
                        st.session_state['cloud_eval_logs'][model_name] = eval_result
                        log(f'⚡ [{model_name}] 云端评估结果已存在，跳过重复计算', 'success')
                    except Exception:
                        st.session_state['cloud_eval_logs'][model_name] = cmd_output
                        log(f'[EVAL] 评估完成', 'info')
                    finally:
                        sftp_read.close()
                else:
                    st.session_state['cloud_eval_logs'][model_name] = cmd_output
                    log(f'[EVAL] 评估完成，结果已存入云端 {remote_eval_log}', 'info')
            else:
                log(f'[INFER] 推理完成', 'info')

            progress_bar.progress((i + 0.5) / len(selected_models))

            log(f'📥 步骤4/4: 下载结果 [{model_name}]...', 'info')

            os.makedirs(local_model_dir, exist_ok=True)

            sftp = ssh.open_sftp()
            try:
                sftp.get(output_nii, local_pred_path)

                pred_data = nib.load(local_pred_path).get_fdata(dtype=np.float32).astype(np.int16)
                st.session_state['pred_results'][model_name] = pred_data
                st.session_state['remote_pred_paths'][model_name] = output_nii
                log(f'✅ {model_name} 完成，已缓存到本地: {local_pred_path}', 'success')

                if gt_file and remote_gt_path and cmd_output:
                    with open(local_eval_path, 'w') as f:
                        f.write(cmd_output)
                    log(f'📥 评估结果已缓存: {local_eval_path}', 'info')

            except FileNotFoundError:
                log(f'❌ {model_name} 预测失败 - 未在云端找到生成的文件: {output_nii}', 'error')
                log(f'[云端日志输出]:\n{cmd_output[-500:]}', 'warning')

            sftp.close()
            progress_bar.progress((i + 1) / len(selected_models))

        progress_bar.progress(1.0)
        ssh.close()

        local_raw_save = save_uploaded_file(raw_file, '_raw.nii.gz')
        st.session_state['remote_raw_file'] = local_raw_save

        if gt_file:
            local_gt_save = save_uploaded_file(gt_file, '_gt.nii.gz')
            st.session_state['remote_gt_file'] = local_gt_save

        log('✅ 所有任务执行完毕！', 'success')
        status.update(label='✅ 所有任务执行完毕！正在渲染结果...', state='complete')

        st.session_state['inference_log'] = log_messages
        st.rerun()

    except Exception as e:
        log(f'❌ 远程任务失败: {str(e)}', 'error')
        st.session_state['inference_log'] = log_messages
        if ssh:
            ssh.close()