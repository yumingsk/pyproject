import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tempfile
import os
import torch
import torch.nn.functional as F
import glob
import gc

# ==========================================
# 🌟 修复 1：彻底消灭 Matplotlib 终端中文报警
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 🌟 修复 2：智能数据降采样特判模块 (防卡死核心)
# ==========================================
def auto_downsample(volume, max_dim=256, is_label=False):
    """自动等比例下采样过大的 3D 数据，拯救显存和 Web 推理时间"""
    shape = volume.shape
    if max(shape) <= max_dim:
        return volume

    scale = max_dim / max(shape)
    new_shape = (max(32, int(shape[0] * scale)),
                 max(32, int(shape[1] * scale)),
                 max(32, int(shape[2] * scale)))

    tensor = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0)
    mode = 'nearest' if is_label else 'trilinear'
    align = None if is_label else False

    resized_tensor = F.interpolate(tensor, size=new_shape, mode=mode, align_corners=align)
    return resized_tensor.squeeze().numpy()


# 导入模型及工具类 (请确保路径与你本地一致)
from models.SKCDF import VNet_Decouple_Attention_ABC
from utils.config import Config
from utils import test_single_case

# --- 基础配置与模型映射 ---
st.set_page_config(page_title="3D医学图像多模型对比工具", layout="wide")

TASK_NAME = 'synapse'
CKPT_DIR = "./ckpts"
RESULT_DIR = os.path.abspath("inference_results")
for d in [CKPT_DIR, RESULT_DIR]:
    if not os.path.exists(d): os.makedirs(d, exist_ok=True)

config = Config(TASK_NAME)

MODEL_DISPLAY_MAP = {
    "vnet_best.pth": "V-Net 深度预测版",
    "skcdf_v1.pth": "SKCDF 注意力模型",
    "baseline.pth": "基础对照模型"
}

LABEL_MAP = {
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


# --- 核心逻辑函数 ---
def get_model_list():
    paths = glob.glob(os.path.join(CKPT_DIR, "*.pth"))
    files = sorted([os.path.basename(p) for p in paths])
    return files


@st.cache_resource(max_entries=2)
def load_model_instance(model_name):
    model = VNet_Decouple_Attention_ABC(n_channels=config.num_channels, n_classes=config.num_cls,
                                        n_filters=config.n_filters, normalization='batchnorm', has_dropout=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(CKPT_DIR, model_name)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"模型 {model_name} 加载失败: {e}")
        return None, None


def get_synapse_cmap():
    colors = [v["color"] for v in LABEL_MAP.values()]
    cmap_colors = colors.copy()
    cmap_colors[0] = (0, 0, 0, 0)
    return mcolors.ListedColormap(cmap_colors)


# ==========================================
# 🌟 修复 3：纯净版画图函数 + HTML统标题渲染
# ==========================================
def render_slice(img_s, seg_s, cmap, alpha, aspect=1.0, show_seg=True):
    """只负责画图，不负责写字，避免长宽比导致的字体缩放畸变"""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(img_s.T, cmap='gray', origin='lower', interpolation='bilinear', aspect=aspect)
    if show_seg and seg_s is not None and np.any(seg_s > 0):
        masked_seg = np.ma.masked_where(seg_s == 0, seg_s)
        ax.imshow(masked_seg.T, cmap=cmap, alpha=alpha, vmin=0, vmax=len(cmap.colors) - 1,
                  origin='lower', interpolation='nearest', aspect=aspect)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def render_title(text):
    """交给浏览器原生渲染的标题，大小绝对统一"""
    st.markdown(f"""
        <div style='text-align: center; background-color: #333333; color: white; 
                    padding: 4px; border-radius: 4px; font-size: 14px; margin-bottom: 5px;'>
            {text}
        </div>
    """, unsafe_allow_html=True)


# --- UI 布局 ---
col_ctrl, col_main = st.columns([1, 4])

with col_ctrl:
    st.subheader("🛠️ 控制面板")

    actual_files = get_model_list()
    if not actual_files:
        st.warning("未在 ckpts 目录找到 .pth 文件")
        selected_display_names = []
    else:
        name_to_file = {MODEL_DISPLAY_MAP.get(f, f): f for f in actual_files}
        display_options = list(name_to_file.keys())
        selected_display_names = st.multiselect(
            "选择预测模型 (支持多选对比)",
            options=display_options,
            default=[display_options[0]] if display_options else None
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

    if raw_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as t:
            t.write(raw_file.getvalue())
            tmp_path = t.name
        nii = nib.load(tmp_path)
        raw_data = nii.get_fdata()
        zooms = nii.header.get_zooms()

        # --- 触发降采样特判 ---
        original_shape = raw_data.shape
        raw_data = auto_downsample(raw_data, max_dim=256, is_label=False)
        dims = raw_data.shape
        if original_shape != dims:
            st.toast(f"⚠️ 原图过于庞大 {original_shape}，已自动降采样至 {dims} 以保障网页流畅度！")

        os.remove(tmp_path)

        st.subheader("🧭 切片导航")
        idx_w = st.slider("矢状面 (Sagittal)", 0, dims[0] - 1, dims[0] // 2)
        idx_h = st.slider("冠状面 (Coronal)", 0, dims[1] - 1, dims[1] // 2)
        idx_d = st.slider("横断面 (Axial)", 0, dims[2] - 1, dims[2] // 2)

        if gt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as t:
                t.write(gt_file.getvalue())
                tmp_gt = t.name
            gt_data = nib.load(tmp_gt).get_fdata()
            # 标签也必须同步降采样
            gt_data = auto_downsample(gt_data, max_dim=256, is_label=True)
            os.remove(tmp_gt)

    run_btn = st.button("🚀 开始批量推理", type="primary", use_container_width=True,
                        disabled=(not raw_file or not selected_display_names))

with col_main:
    st.title("医学图像分割对比分析平台")

    legend_cols = st.columns(7)
    for idx, (en, info) in enumerate(list(LABEL_MAP.items())[1:]):
        with legend_cols[idx % 7]:
            st.markdown(
                f"<div style='border-left: 4px solid {info['color']}; padding-left:4px; font-size:11px;'>{info['zh']}</div>",
                unsafe_allow_html=True)

    if raw_file and raw_data is not None:
        disp_bg = np.clip(raw_data, -125, 275)
        disp_bg = (disp_bg - disp_bg.min()) / (disp_bg.max() - disp_bg.min() + 1e-8)
        cmap = get_synapse_cmap()

        # --- 推理逻辑 ---
        if run_btn:
            st.session_state["pred_results"] = {}
            status_text = st.empty()
            progress_bar = st.progress(0)

            status_text.info("🔄 正在进行数据标准化...")
            img_trans = raw_data.transpose(2, 0, 1)
            img_norm = np.clip(img_trans, -75, 275)
            img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
            img_tensor = img_norm.astype(np.float32)

            total_models = len(selected_display_names)

            for i, d_name in enumerate(selected_display_names):
                f_name = name_to_file[d_name]
                status_text.warning(f"🚀 正在加载模型 [{d_name}] ...")

                model_inst, device = load_model_instance(f_name)

                if model_inst:
                    status_text.warning(
                        f"🧠 模型 [{d_name}] 推理中...\n(步长 {stride_xy}x{stride_xy}x{stride_z}，由于 3D 计算量极大，请耐心等待数十秒)")

                    with torch.no_grad():
                        mask, _ = test_single_case(
                            model_inst, img_tensor, stride_xy, stride_z,
                            config.patch_size, config.num_cls
                        )
                        st.session_state["pred_results"][d_name] = mask.transpose(1, 2, 0)

                    del mask
                    torch.cuda.empty_cache()
                    gc.collect()

                progress_bar.progress((i + 1) / total_models)

            status_text.success("✅ 所有推理任务已完成！结果已渲染。")

        # --- 视图渲染 (图文分离，彻底解决大小不一) ---
        with st.expander("🖼️ 原始 CT 图像", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                render_title("原始图像 - 矢状面 (Sagittal)")
                st.pyplot(render_slice(disp_bg[idx_w, :, :], None, cmap, alpha, zooms[2] / zooms[1], False))
            with c2:
                render_title("原始图像 - 冠状面 (Coronal)")
                st.pyplot(render_slice(disp_bg[:, idx_h, :], None, cmap, alpha, zooms[2] / zooms[0], False))
            with c3:
                render_title("原始图像 - 横断面 (Axial)")
                st.pyplot(render_slice(disp_bg[:, :, idx_d], None, cmap, alpha, zooms[1] / zooms[0], False))

        if gt_data is not None:
            with st.expander("✅ 专家标注 (Ground Truth)", expanded=True):
                g1, g2, g3 = st.columns(3)
                with g1:
                    render_title("专家标注 - 矢状面")
                    st.pyplot(
                        render_slice(disp_bg[idx_w, :, :], gt_data[idx_w, :, :], cmap, alpha, zooms[2] / zooms[1]))
                with g2:
                    render_title("专家标注 - 冠状面")
                    st.pyplot(
                        render_slice(disp_bg[:, idx_h, :], gt_data[:, idx_h, :], cmap, alpha, zooms[2] / zooms[0]))
                with g3:
                    render_title("专家标注 - 横断面")
                    st.pyplot(
                        render_slice(disp_bg[:, :, idx_d], gt_data[:, :, idx_d], cmap, alpha, zooms[1] / zooms[0]))

        saved_preds = st.session_state.get("pred_results", {})
        for d_name in selected_display_names:
            if d_name in saved_preds:
                st.markdown(f"#### 🤖 模型预测: {d_name}")
                p_mask = saved_preds[d_name]
                p1, p2, p3 = st.columns(3)
                with p1:
                    render_title("预测模型 - 矢状面")
                    st.pyplot(render_slice(disp_bg[idx_w, :, :], p_mask[idx_w, :, :], cmap, alpha, zooms[2] / zooms[1]))
                with p2:
                    render_title("预测模型 - 冠状面")
                    st.pyplot(render_slice(disp_bg[:, idx_h, :], p_mask[:, idx_h, :], cmap, alpha, zooms[2] / zooms[0]))
                with p3:
                    render_title("预测模型 - 横断面")
                    st.pyplot(render_slice(disp_bg[:, :, idx_d], p_mask[:, :, idx_d], cmap, alpha, zooms[1] / zooms[0]))
                st.divider()
    else:
        st.info("💡 请在左侧上传 NIfTI 格式的 CT 影像数据以开始分析。")