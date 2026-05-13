import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tempfile
import os
import io
import torch
import torch.nn.functional as F
import glob
import gc

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def auto_downsample(volume, max_dim=192, min_dim=64, is_label=False):
    max_dim_for_memory = 192
    shape = volume.shape
    if max(shape) <= max_dim and min(shape) >= min_dim:
        return volume

    scale_down = max_dim_for_memory / max(shape) if max(shape) > max_dim_for_memory else 1.0
    scale_up = min_dim / min(shape) if min(shape) < min_dim else 1.0
    
    if scale_up > 1.0:
        scale = scale_up
    elif scale_down < 1.0:
        scale = scale_down
    else:
        return volume

    new_shape = (max(min_dim, int(shape[0] * scale)),
                 max(min_dim, int(shape[1] * scale)),
                 max(min_dim, int(shape[2] * scale)))

    tensor = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)
    mode = 'nearest' if is_label else 'trilinear'
    align = None if is_label else False

    resized_tensor = F.interpolate(tensor, size=new_shape, mode=mode, align_corners=align)
    return resized_tensor.squeeze().numpy()


from models.SKCDF import VNet_Decouple_Attention_ABC
from models.vnet import VNet
from models.DHC.vnet_dst import VNet_Decoupled
from models.DHC.vnet_flat import VNet as VNet_DHC
from utils.config import Config
from utils import test_single_case, test_single_case_AB

st.set_page_config(page_title="3D医学图像多模型对比工具", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "inference_results")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR, exist_ok=True)

LABEL_MAP_SYNAPSE = {
    "Background": {"zh": "背景", "color": "#000000"},
    "Spleen": {"zh": "脾脏", "color": "#008000"},
    "R-Kidney": {"zh": "右肾", "color": "#00FF00"},
    "L-Kidney": {"zh": "左肾", "color": "#FF0000"},
    "Gallbladder": {"zh": "胆囊", "color": "#0000FF"},
    "Esophagus": {"zh": "食道", "color": "#1E90FF"},
    "Liver": {"zh": "肝脏", "color": "#FF69B4"},
    "Stomach": {"zh": "胃", "color": "#8B00FF"},
    "Aorta": {"zh": "主动脉", "color": "#FF8C00"},
    "IVC": {"zh": "下腔静脉", "color": "#A52A2A"},
    "Veins": {"zh": "门静脉/脾静脉", "color": "#8B4513"},
    "Pancreas": {"zh": "胰腺", "color": "#FF00FF"},
    "R-Adrenal": {"zh": "右肾上腺", "color": "#00FFFF"},
    "L-Adrenal": {"zh": "左肾上腺", "color": "#FFFF00"}
}

LABEL_MAP_AMOS = {
    "Background": {"zh": "背景", "color": "#000000"},
    "Spleen": {"zh": "脾脏", "color": "#008000"},
    "R-Kidney": {"zh": "右肾", "color": "#00FF00"},
    "L-Kidney": {"zh": "左肾", "color": "#FF0000"},
    "Gallbladder": {"zh": "胆囊", "color": "#0000FF"},
    "Esophagus": {"zh": "食道", "color": "#1E90FF"},
    "Liver": {"zh": "肝脏", "color": "#FF69B4"},
    "Stomach": {"zh": "胃", "color": "#8B00FF"},
    "Aorta": {"zh": "主动脉", "color": "#FF8C00"},
    "IVC": {"zh": "下腔静脉", "color": "#A52A2A"},
    "Pancreas": {"zh": "胰腺", "color": "#FF00FF"},
    "R-Adrenal": {"zh": "右肾上腺", "color": "#00FFFF"},
    "L-Adrenal": {"zh": "左肾上腺", "color": "#FFFF00"},
    "Duodenum": {"zh": "十二指肠", "color": "#FF6347"},
    "Bladder": {"zh": "膀胱", "color": "#40E0D0"},
    "Prostate/Uterus": {"zh": "前列腺/子宫", "color": "#DA70D6"}
}

DATASET_CONFIG = {
    "Synapse": {
        "task": "synapse",
        "label_map": LABEL_MAP_SYNAPSE,
        "ckpt_dir": os.path.join(BASE_DIR, "ckpts", "synapse"),
        "description": "MICCAI 2015 · 13个前景类别 · 30例CT扫描"
    },
    "AMOS": {
        "task": "amos",
        "label_map": LABEL_MAP_AMOS,
        "ckpt_dir": os.path.join(BASE_DIR, "ckpts", "amos"),
        "description": "MICCAI 2022 · 15个前景类别 · 360例CT扫描"
    }
}

for ds_cfg in DATASET_CONFIG.values():
    os.makedirs(ds_cfg["ckpt_dir"], exist_ok=True)


def get_model_list(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    files = sorted([os.path.basename(p) for p in paths])
    return files


@st.cache_resource(max_entries=2)
def load_model_instance(model_name, _ckpt_dir, num_cls, n_filters=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(_ckpt_dir, model_name)
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'DHC' in model_name or 'dhc' in model_name:
            if 'A' in state_dict and 'B' in state_dict:
                model_A = VNet_DHC(n_channels=1, n_classes=num_cls, n_filters=n_filters, normalization='batchnorm', has_dropout=False)
                model_B = VNet_DHC(n_channels=1, n_classes=num_cls, n_filters=n_filters, normalization='batchnorm', has_dropout=False)
                
                model_A.load_state_dict(state_dict['A'])
                model_B.load_state_dict(state_dict['B'])
                
                model_A.to(device).eval()
                model_B.to(device).eval()
                
                return (model_A, model_B), device
            else:
                st.error(f"模型 {model_name} 不是有效的DHC双模型格式")
                return None, None
        elif 'vnet' in model_name.lower() and 'attention' not in model_name.lower():
            model = VNet(n_channels=1, n_classes=num_cls, n_filters=n_filters, normalization='batchnorm')
            model.load_state_dict(state_dict)
            model.to(device).eval()
            return model, device
        else:
            model = VNet_Decouple_Attention_ABC(n_channels=1, n_classes=num_cls,
                                                n_filters=n_filters, normalization='batchnorm', has_dropout=False)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            return model, device
    except Exception as e:
        st.error(f"模型 {model_name} 加载失败: {e}")
        return None, None


def get_cmap(label_map):
    colors = [v["color"] for v in label_map.values()]
    cmap_colors = colors.copy()
    cmap_colors[0] = (0, 0, 0, 0)
    return mcolors.ListedColormap(cmap_colors)


def render_slice(img_s, seg_s, cmap, alpha, aspect=1.0, show_seg=True):
    h, w = img_s.shape
    
    base_size = 4.0
    fig_height = base_size
    fig_width = base_size * max(1.0, (w / h) * aspect)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=140)
    ax.imshow(img_s.T, cmap='gray', origin='lower', interpolation='bilinear', aspect='auto')
    if show_seg and seg_s is not None and np.any(seg_s > 0):
        masked_seg = np.ma.masked_where(seg_s == 0, seg_s)
        ax.imshow(masked_seg.T, cmap=cmap, alpha=alpha, vmin=0, vmax=len(cmap.colors) - 1,
                  origin='lower', interpolation='nearest', aspect='auto')
    ax.axis('off')
    plt.tight_layout(pad=0.02)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.01, dpi=140)
    buf.seek(0)
    plt.close(fig)
    return buf


def render_title(text):
    st.markdown(f"""
        <div style='text-align: center; background-color: #333333; color: white; 
                    padding: 4px; border-radius: 4px; font-size: 14px; margin-bottom: 5px;'>
            {text}
        </div>
    """, unsafe_allow_html=True)


# --- UI 布局 ---
col_ctrl, col_main = st.columns([1, 4])

with col_ctrl:
    st.subheader("控制面板")
    
    inference_mode = st.radio(
        "🖥️ 推理模式",
        options=["本地CPU", "远程GPU"],
        index=0,
        horizontal=True,
        help="本地CPU: 在本机运行，速度快但精度受限 | 远程GPU: 在远程服务器运行，精度高但需要网络连接"
    )
    
    st.session_state['inference_mode'] = inference_mode
    
    if inference_mode == "远程GPU":
        st.divider()
        from utils.remote_inference import render_gpu_inference_ui
        render_gpu_inference_ui()
        dataset_choice = "Synapse"
        ds_cfg = DATASET_CONFIG[dataset_choice]
        label_map = ds_cfg["label_map"]
        raw_file = None
        selected_display_names = []
    else:
        dataset_choice = st.radio(
            "📂 选择数据集",
            options=["Synapse", "AMOS"],
            index=0,
            horizontal=True,
            help="Synapse: 13个前景类别 | AMOS: 15个前景类别(含十二指肠、膀胱、前列腺/子宫)"
        )

        ds_cfg = DATASET_CONFIG[dataset_choice]
        task_name = ds_cfg["task"]
        label_map = ds_cfg["label_map"]
        ckpt_dir = ds_cfg["ckpt_dir"]
        config = Config(task_name)

        st.caption(f"{ds_cfg['description']}")

        st.divider()

        actual_files = get_model_list(ckpt_dir)
        if not actual_files:
            st.warning(f"未在 {ckpt_dir} 目录找到 .pth 文件")
            selected_display_names = []
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-weight: 600;">选择预测模型 (支持多选对比)</span>
                <span title="命名规则：方法名_数据集_训练数据百分比.pth&#10;例如：BCP-SCKDF_synapse_20p.pth&#10;- 方法：BCP-SCKDF&#10;- 数据集：synapse&#10;- 训练数据：20%" style="cursor: help; font-size: 16px;">❓</span>
            </div>
            """, unsafe_allow_html=True)
            
            selected_display_names = st.multiselect(
                "选择预测模型 (支持多选对比)",
                options=actual_files,
                default=[actual_files[0]] if actual_files else None,
                label_visibility="collapsed"
            )

        st.divider()
        st.markdown("**推理速度/精度调节**")
        speed_mode = st.radio(
            "滑动窗口步长 (影响推理耗时):",
            options=["快速预览 (步长 64x64x32)", "均衡模式 (步长 48x48x24)", "极限精度 (步长 32x32x16 - 极慢)"],
            index=0,
            help="步长越小，推理越准但耗时倍增。大图片请务必选择'快速预览'！"
        )

        if "快速" in speed_mode:
            stride_xy, stride_z = 64, 32
        elif "均衡" in speed_mode:
            stride_xy, stride_z = 48, 24
        else:
            stride_xy, stride_z = 32, 16

        st.divider()
        raw_file = st.file_uploader("1. 上传 Raw CT (.nii.gz)", type=['nii', 'nii.gz'])
        gt_file = st.file_uploader("2. 上传 Truth (.nii.gz) [可选]", type=['nii', 'nii.gz'])
        alpha = st.slider("分割层透明度", 0.0, 1.0, 0.5)

        raw_data, gt_data = None, None
        zooms, dims = (1., 1., 1.), (1, 1, 1)
        min_input_dim = 64

        if raw_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as t:
                t.write(raw_file.getvalue())
                tmp_path = t.name
            nii = nib.load(tmp_path)
            raw_data = nii.get_fdata(dtype=np.float32)
            zooms = nii.header.get_zooms()

            original_shape = raw_data.shape
            raw_data = auto_downsample(raw_data, max_dim=192, min_dim=min_input_dim, is_label=False)
            dims = raw_data.shape
            
            if original_shape != dims:
                if min(original_shape) < min_input_dim:
                    st.toast(f"原图尺寸过小 {original_shape}，已自动上采样至 {dims} 以满足模型输入要求！")
                else:
                    st.toast(f"原图过于庞大 {original_shape}，已自动降采样至 {dims} 以保障网页流畅度！")

            os.remove(tmp_path)

        st.subheader("切片导航")
# 增加判断：只有在 dims 被正确赋值（即上传了CT）后，才渲染滑动条
        if dims[0] > 1 and dims[1] > 1 and dims[2] > 1:
            idx_w = st.slider("矢状面 (Sagittal)", 0, dims[0] - 1, dims[0] // 2)
            idx_h = st.slider("冠状面 (Coronal)", 0, dims[1] - 1, dims[1] // 2)
            idx_d = st.slider("横断面 (Axial)", 0, dims[2] - 1, dims[2] // 2)
        else:
    # 占位默认值，防止后续代码引用变量报错
            idx_w, idx_h, idx_d = 0, 0, 0
        st.info("📌 请先上传 Raw CT 图像以启用切片导航")

        if gt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as t:
                t.write(gt_file.getvalue())
                tmp_gt = t.name
            gt_data = nib.load(tmp_gt).get_fdata(dtype=np.float32)
            gt_data = auto_downsample(gt_data, max_dim=192, min_dim=min_input_dim, is_label=True)
            os.remove(tmp_gt)

        run_btn = st.button("开始推理", type="primary", use_container_width=True,
                            disabled=(not raw_file or not selected_display_names))

with col_main:
    if st.session_state.get('inference_mode') == "远程GPU":
        st.title("医学图像分割对比分析平台 - 远程GPU模式")
        
        remote_raw_file = st.session_state.get('remote_raw_file')
        remote_gt_file = st.session_state.get('remote_gt_file')
        remote_pred_results = st.session_state.get("pred_results", {})
        
        if remote_raw_file is not None:
            raw_data = nib.load(remote_raw_file).get_fdata(dtype=np.float32)
            dims = raw_data.shape
            disp_bg = np.clip(raw_data, -125, 275)
            disp_bg = (disp_bg - disp_bg.min()) / (disp_bg.max() - disp_bg.min() + 1e-8)
            
            label_map = LABEL_MAP_SYNAPSE
            cmap = get_cmap(label_map)
            
            num_legend_cols = 7
            legend_cols = st.columns(num_legend_cols)
            for idx, (en, info) in enumerate(list(label_map.items())[1:]):
                with legend_cols[idx % num_legend_cols]:
                    st.markdown(
                        f"<div style='border-left: 4px solid {info['color']}; padding-left:4px; font-size:11px;'>{info['zh']}</div>",
                        unsafe_allow_html=True)
            
            if dims[0] > 1 and dims[1] > 1 and dims[2] > 1:
                zooms = (1., 1., 1.)
                
                st.subheader("切片导航")
                idx_w = st.slider("矢状面 (Sagittal)", 0, dims[0] - 1, dims[0] // 2)
                idx_h = st.slider("冠状面 (Coronal)", 0, dims[1] - 1, dims[1] // 2)
                idx_d = st.slider("横断面 (Axial)", 0, dims[2] - 1, dims[2] // 2)
                
                with st.expander("原始 CT 图像", expanded=True):
                    c1, c2, c3 = st.columns([1.5, 1.5, 1])
                    with c1:
                        render_title("矢状面 (Sagittal)")
                        st.image(render_slice(disp_bg[idx_w, :, :], None, cmap, 0.5, zooms[2] / zooms[1], False))
                    with c2:
                        render_title("冠状面 (Coronal)")
                        st.image(render_slice(disp_bg[:, idx_h, :], None, cmap, 0.5, zooms[2] / zooms[0], False))
                    with c3:
                        render_title("横断面 (Axial)")
                        st.image(render_slice(disp_bg[:, :, idx_d], None, cmap, 0.5, zooms[1] / zooms[0], False))
                
                gt_data = None
                if remote_gt_file:
                    gt_data = nib.load(remote_gt_file).get_fdata(dtype=np.float32)
                    if gt_data.shape != raw_data.shape:
                        if gt_data.shape == (raw_data.shape[2], raw_data.shape[0], raw_data.shape[1]):
                            gt_data = gt_data.transpose(1, 2, 0)
                        elif gt_data.shape == (raw_data.shape[1], raw_data.shape[2], raw_data.shape[0]):
                            gt_data = gt_data.transpose(2, 0, 1)
                    with st.expander("专家标注", expanded=True):
                        g1, g2, g3 = st.columns([1.5, 1.5, 1])
                        with g1:
                            render_title("矢状面")
                            st.image(render_slice(disp_bg[idx_w, :, :], gt_data[idx_w, :, :], cmap, 0.5, zooms[2] / zooms[1]))
                        with g2:
                            render_title("冠状面")
                            st.image(render_slice(disp_bg[:, idx_h, :], gt_data[:, idx_h, :], cmap, 0.5, zooms[2] / zooms[0]))
                        with g3:
                            render_title("横断面")
                            st.image(render_slice(disp_bg[:, :, idx_d], gt_data[:, :, idx_d], cmap, 0.5, zooms[1] / zooms[0]))
                
                saved_preds = remote_pred_results
                selected_display_names = list(saved_preds.keys())
                
                for d_name in selected_display_names:
                    if d_name in saved_preds:
                        p_mask = saved_preds[d_name]

                        if p_mask.shape != raw_data.shape:
                            if p_mask.shape == (raw_data.shape[2], raw_data.shape[0], raw_data.shape[1]):
                                p_mask = p_mask.transpose(1, 2, 0)
                            elif p_mask.shape == (raw_data.shape[1], raw_data.shape[2], raw_data.shape[0]):
                                p_mask = p_mask.transpose(2, 0, 1)
                        with st.expander(f"模型预测: {d_name}", expanded=True):
                            p1, p2, p3 = st.columns([1.5, 1.5, 1])
                            with p1:
                                render_title("矢状面")
                                st.image(render_slice(disp_bg[idx_w, :, :], p_mask[idx_w, :, :], cmap, 0.5, zooms[2] / zooms[1]))
                            with p2:
                                render_title("冠状面")
                                st.image(render_slice(disp_bg[:, idx_h, :], p_mask[:, idx_h, :], cmap, 0.5, zooms[2] / zooms[0]))
                            with p3:
                                render_title("横断面")
                                st.image(render_slice(disp_bg[:, :, idx_d], p_mask[:, :, idx_d], cmap, 0.5, zooms[1] / zooms[0]))

                if saved_preds and gt_data is not None:
                    st.divider()
                    st.subheader("📊 云端评价指标 (全分辨率高精度)")

                    import pandas as pd
                    import re

                    # 🛡️ 兼容不同的变量名，防止拿不到数据
                    cloud_logs = st.session_state.get('eval_results') or st.session_state.get('cloud_eval_logs') or {}

                    if cloud_logs:
                        dice_data = []
                        asd_data = []

                        cloud_to_zh = {
                            'spleen': '脾脏', 'right_kidney': '右肾', 'left_kidney': '左肾',
                            'gallbladder': '胆囊', 'esophagus': '食道', 'liver': '肝脏',
                            'stomach': '胃', 'aorta': '主动脉', 'ivc': '下腔静脉',
                            'portal_vein': '门静脉/脾静脉', 'pancreas': '胰腺',
                            'adrenal_gland_right': '右肾上腺', 'adrenal_gland_left': '左肾上腺',
                            'duodenum': '十二指肠', 'bladder': '膀胱', 'prostate_uterus': '前列腺/子宫',
                            'average': 'Average'  # 使用小写做 key 增加鲁棒性
                        }

                        for m_name in selected_display_names:
                            if m_name in cloud_logs:
                                eval_text = cloud_logs[m_name]
                                dice_row = {'Model': m_name}
                                asd_row = {'Model': m_name}

                                for line in eval_text.split('\n'):
                                    # 统一转为小写处理，防止匹配失败
                                    line_lower = line.lower()
                                    if ':' in line_lower and 'dice=' in line_lower and 'asd=' in line_lower:
                                        organ_eng = line_lower.split(':')[0].strip()

                                        # 正则提取数字
                                        dice_match = re.search(r'dice=\s*([0-9.]+)', line_lower)
                                        asd_match = re.search(r'asd=\s*([0-9.]+|nan)', line_lower)

                                        if dice_match and asd_match and organ_eng in cloud_to_zh:
                                            zh_name = cloud_to_zh[organ_eng]
                                            dice_row[zh_name] = float(dice_match.group(1))

                                            asd_str = asd_match.group(1)
                                            asd_row[zh_name] = float('nan') if asd_str == 'nan' else float(asd_str)

                                # 只要解析出至少一个器官的数据，就加入表格
                                if len(dice_row) > 1:
                                    dice_data.append(dice_row)
                                    asd_data.append(asd_row)

                        if dice_data:
                            dice_df = pd.DataFrame(dice_data)
                            asd_df = pd.DataFrame(asd_data)

                            # 调整列顺序，确保 Average 永远在最后一列
                            if 'Average' in dice_df.columns:
                                cols = [c for c in dice_df.columns if c != 'Average'] + ['Average']
                                dice_df = dice_df[cols]
                                asd_df = asd_df[cols]


                            def bold_best(df, is_max=True):
                                df_out = df.copy()
                                cols = [c for c in df.columns if c != 'Model']
                                for col in cols:
                                    if df[col].isnull().all():
                                        continue
                                    best_val = df[col].max() if is_max else df[col].min()
                                    df_out[col] = df[col].apply(
                                        lambda x: f"**{x:.2f}**" if x == best_val else (
                                            f"{x:.2f}" if pd.notnull(x) else "NaN")
                                    )
                                return df_out


                            styled_dice = bold_best(dice_df, is_max=True)
                            styled_asd = bold_best(asd_df, is_max=False)

                            st.markdown("**🎯 DICE (%)** - 数值越大越好")
                            st.markdown(styled_dice.to_markdown(index=False))

                            st.markdown("")
                            st.markdown("**📏 ASD (mm)** - 数值越小越好")
                            st.markdown(styled_asd.to_markdown(index=False))

                            st.markdown("")
                            st.markdown("💡 **说明**: 以上结果由云端 GPU 计算得出。**加粗** 表示当前对比模型中的最优值。")

                            with st.expander("📋 查看云端原始打分日志"):
                                for m_name in selected_display_names:
                                    if m_name in cloud_logs:
                                        st.text(f"--- 模型: {m_name} ---")
                                        st.code(cloud_logs[m_name], language='text')
                        else:
                            st.error("⚠️ 未能从云端返回的文本中解析出表格数据，请查看下方原始日志：")
                            with st.expander("📋 原始打分日志", expanded=True):
                                for m_name in selected_display_names:
                                    if m_name in cloud_logs:
                                        st.code(cloud_logs[m_name], language='text')
                    else:
                        st.info("🔄 暂无云端评估结果。请点击【开始远程GPU预测】。")
        else:
            st.info("请在左侧控制面板中配置远程GPU连接并上传数据进行推理。")
    else:
        st.title("医学图像分割对比分析平台")

        num_legend_cols = 8 if dataset_choice == "AMOS" else 7
        legend_cols = st.columns(num_legend_cols)
        for idx, (en, info) in enumerate(list(label_map.items())[1:]):
            with legend_cols[idx % num_legend_cols]:
                st.markdown(
                    f"<div style='border-left: 4px solid {info['color']}; padding-left:4px; font-size:11px;'>{info['zh']}</div>",
                    unsafe_allow_html=True)

        if raw_file and raw_data is not None:
            disp_bg = np.clip(raw_data, -125, 275)
            disp_bg = (disp_bg - disp_bg.min()) / (disp_bg.max() - disp_bg.min() + 1e-8)
            cmap = get_cmap(label_map)

            if run_btn:
                st.session_state["pred_results"] = {}
                status_text = st.empty()
                progress_bar = st.progress(0)

                status_text.info("正在进行数据标准化...")
                img_trans = raw_data.transpose(2, 0, 1)
                img_norm = np.clip(img_trans, -75, 275)
                img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
                img_tensor = img_norm.astype(np.float32)

                min_sizes = config.patch_size
                current_shape = img_tensor.shape
                need_resize = False
                new_shape = list(current_shape)
                
                for idx in range(3):
                    if current_shape[idx] < min_sizes[idx]:
                        new_shape[idx] = min_sizes[idx]
                        need_resize = True
                
                if need_resize:
                    status_text.info(f"输入尺寸 {current_shape} 小于模型要求 {min_sizes}，正在上采样...")
                    img_tensor = auto_downsample(
                        img_tensor.transpose(1, 2, 0), 
                        max_dim=max(new_shape), 
                        min_dim=max(min_sizes), 
                        is_label=False
                    ).transpose(2, 0, 1)
                    status_text.info(f"已上采样至 {img_tensor.shape}")

                total_models = len(selected_display_names)

                for i, f_name in enumerate(selected_display_names):
                    status_text.warning(f"正在加载模型 [{f_name}] ...")

                    model_inst, device = load_model_instance(
                        f_name, _ckpt_dir=ckpt_dir, num_cls=config.num_cls, n_filters=config.n_filters
                    )

                    if model_inst:
                        status_text.warning(
                            f"模型 [{f_name}] 推理中...\n(步长 {stride_xy}x{stride_xy}x{stride_z}，由于 3D 计算量极大，请耐心等待数十秒)")

                        with torch.no_grad():
                            if isinstance(model_inst, tuple):
                                model_A, model_B = model_inst
                                mask, _ = test_single_case_AB(
                                    model_A, model_B, img_tensor, stride_xy, stride_z,
                                    config.patch_size, config.num_cls
                                )
                            else:
                                mask, _ = test_single_case(
                                    model_inst, img_tensor, stride_xy, stride_z,
                                    config.patch_size, config.num_cls
                                )
                            st.session_state["pred_results"][f_name] = mask.transpose(1, 2, 0)

                            del mask
                            torch.cuda.empty_cache()
                            gc.collect()

                    progress_bar.progress((i + 1) / total_models)

                status_text.success("所有推理任务已完成！结果已渲染。")

            with st.expander("原始 CT 图像", expanded=True):
                c1, c2, c3 = st.columns([1.5, 1.5, 1])
                with c1:
                    render_title("原始图像 - 矢状面 (Sagittal)")
                    st.image(render_slice(disp_bg[idx_w, :, :], None, cmap, alpha, zooms[2] / zooms[1], False))
                with c2:
                    render_title("原始图像 - 冠状面 (Coronal)")
                    st.image(render_slice(disp_bg[:, idx_h, :], None, cmap, alpha, zooms[2] / zooms[0], False))
                with c3:
                    render_title("原始图像 - 横断面 (Axial)")
                    st.image(render_slice(disp_bg[:, :, idx_d], None, cmap, alpha, zooms[1] / zooms[0], False))

            if gt_data is not None:
                with st.expander("专家标注", expanded=True):
                    g1, g2, g3 = st.columns([1.5, 1.5, 1])
                    with g1:
                        render_title("专家标注 - 矢状面")
                        st.image(
                            render_slice(disp_bg[idx_w, :, :], gt_data[idx_w, :, :], cmap, alpha, zooms[2] / zooms[1]))
                    with g2:
                        render_title("专家标注 - 冠状面")
                        st.image(
                            render_slice(disp_bg[:, idx_h, :], gt_data[:, idx_h, :], cmap, alpha, zooms[2] / zooms[0]))
                    with g3:
                        render_title("专家标注 - 横断面")
                        st.image(
                            render_slice(disp_bg[:, :, idx_d], gt_data[:, :, idx_d], cmap, alpha, zooms[1] / zooms[0]))

            saved_preds = st.session_state.get("pred_results", {})
            
            for d_name in selected_display_names:
                if d_name in saved_preds:
                    with st.expander(f"模型预测: {d_name}", expanded=True):
                        p_mask = saved_preds[d_name]
                        p1, p2, p3 = st.columns([1.5, 1.5, 1])
                        with p1:
                            render_title("预测模型 - 矢状面")
                            st.image(render_slice(disp_bg[idx_w, :, :], p_mask[idx_w, :, :], cmap, alpha, zooms[2] / zooms[1]))
                        with p2:
                            render_title("预测模型 - 冠状面")
                            st.image(render_slice(disp_bg[:, idx_h, :], p_mask[:, idx_h, :], cmap, alpha, zooms[2] / zooms[0]))
                        with p3:
                            render_title("预测模型 - 横断面")
                            st.image(render_slice(disp_bg[:, :, idx_d], p_mask[:, :, idx_d], cmap, alpha, zooms[1] / zooms[0]))

            if saved_preds and gt_data is not None:
                st.divider()
                st.subheader("📊 评价指标")

                from utils.evaluation_metrics import evaluate_multiple_models_with_highlight

                label_names = {}
                for en, info in label_map.items():
                    idx = list(label_map.keys()).index(en)
                    label_names[idx] = info['zh']

                # =========================================================
                # 🛡️ 终极防御：创建一个全新的字典，绝不触碰和污染 session_state
                # =========================================================
                safe_eval_preds = {}
                for m_name, pred in saved_preds.items():
                    temp_pred = pred  # 拿出一个副本
                    if temp_pred.shape != gt_data.shape:
                        # (147, 512, 512) -> (512, 512, 147)
                        if temp_pred.shape == (gt_data.shape[2], gt_data.shape[0], gt_data.shape[1]):
                            temp_pred = temp_pred.transpose(1, 2, 0)
                        elif temp_pred.shape == (gt_data.shape[1], gt_data.shape[2], gt_data.shape[0]):
                            temp_pred = temp_pred.transpose(2, 0, 1)
                        elif temp_pred.shape == (gt_data.shape[2], gt_data.shape[1], gt_data.shape[0]):
                            temp_pred = temp_pred.transpose(2, 1, 0)

                    # 把对齐好的形状放进新字典
                    safe_eval_preds[m_name] = temp_pred
                # =========================================================

                # ⚠️ 关键点：这里传入的是 safe_eval_preds，而不是 saved_preds！
                result = evaluate_multiple_models_with_highlight(safe_eval_preds, gt_data, 14, label_names)

                st.markdown("**🎯 DICE (%)** - 数值越大越好")
                st.markdown(result['dice_df'].to_markdown(index=False))

                st.markdown("")
                st.markdown("**📏 ASD (mm)** - 数值越小越好")
                st.markdown(result['asd_df'].to_markdown(index=False))

                st.markdown("")
                st.markdown("💡 **说明**: **加粗** 表示该器官的最优值。DICE最高为最优，ASD最低为最优。")
                
                with st.expander("📋 器官缩写对照表"):
                    organ_mapping = result['organ_mapping']
                    mapping_text = "| 缩写 | 全称 |\n|------|------|\n"
                    for abbr, full_name in organ_mapping.items():
                        mapping_text += f"| {abbr} | {full_name} |\n"
                    st.markdown(mapping_text)
        else:
            st.info("请在左侧上传 NIfTI 格式的 CT 影像数据以开始分析。")
