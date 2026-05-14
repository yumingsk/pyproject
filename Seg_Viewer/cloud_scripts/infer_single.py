import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')

from models.vnet import VNet
from models.SKCDF import VNet_Decouple_Attention_ABC
from models.DHC.vnet_dst import VNet_Decoupled
from models.DHC.vnet_flat import VNet as VNet_DHC  # ⭐ 本地CPU用的类！
from utils import test_single_case, test_single_case_AB, test_single_case_AB_synapse, Config  # ⭐ 添加synapse版本！

def infer_with_test_logic(input_path, output_path, model_path, task='synapse'):
    print("=" * 60)
    print("[START] Single file inference (Forced Training Scale)")
    print("=" * 60)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = Config(task)
    
    # 1. 读取原图
    itk_img = sitk.ReadImage(input_path)
    image = sitk.GetArrayFromImage(itk_img).astype(np.float32) # 原生 (Z, Y, X)
    orig_shape = image.shape
    print(f"\n[IMAGE] Original Physical Shape (Z,Y,X): {orig_shape}")
    
    # 2. 🚨 核心修复：强制缩放到模型训练时的“近视”尺寸 (80, 160, 160) 🚨
    # 完美复刻 preprocess.py 里的插值公式
    resize_shape = (
        config.patch_size[0] + config.patch_size[0] // 4,
        config.patch_size[1] + config.patch_size[1] // 4,
        config.patch_size[2] + config.patch_size[2] // 4
    )
    print(f"[RESIZE] Scaling down to model's training scale: {resize_shape}")
    
    tensor_img = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    tensor_resized = F.interpolate(tensor_img, size=resize_shape, mode='trilinear', align_corners=False)
    img_resized = tensor_resized.squeeze().numpy()
    
    # 3. 归一化 (严格遵守 data_loaders.py 的 -75 到 275)
    img_norm = np.clip(img_resized, -75, 275) 
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    
    print(f"\n[INFERENCE] Running sliding window at scale {resize_shape}...")
    
    # 初始化模型
    model_name = os.path.basename(model_path).lower()
    need_restore = True  # 默认需要尺寸恢复（SKCDF/VNet）
    
    # ========== DHC双模型处理（⭐ 完全复刻 cloud2/test.py + read_data 的正确逻辑！）==========
    if 'dhc' in model_name:
        need_restore = False  # DHC不需要尺寸恢复
        print(f"\n[MODEL] Loading DHC dual-model ensemble: {model_path}")
        
        state_dict = torch.load(model_path, map_location='cuda', weights_only=False)
        
        if 'A' in state_dict and 'B' in state_dict:
            # ⚠️ 修复1: 使用标准VNet（与cloud2/test.py:159-165一致），不是VNet_DHC！
            model_A = VNet(n_channels=1, n_classes=config.num_cls, n_filters=config.n_filters,
                           normalization='batchnorm', has_dropout=False)
            model_B = VNet(n_channels=1, n_classes=config.num_cls, n_filters=config.n_filters,
                           normalization='batchnorm', has_dropout=False)
            
            try:
                model_A.load_state_dict(state_dict['A'])
                model_B.load_state_dict(state_dict['B'])
                print("[MODEL] ✓ DHC ensemble loaded (exact match)")
            except Exception as e:
                print(f"[WARN] Exact load failed: {str(e)[:150]}")
                print("[MODEL] Trying with strict=False...")
                try:
                    model_A.load_state_dict(state_dict['A'], strict=False)
                    model_B.load_state_dict(state_dict['B'], strict=False)
                    print("[MODEL] ✓ DHC ensemble loaded (partial match)")
                except Exception as e2:
                    raise RuntimeError(f"DHC模型加载失败: {str(e2)[:200]}")
            
            model_A.cuda().eval()
            model_B.cuda().eval()
            
            print("[MODEL] ✓ DHC ensemble ready (Model A + Model B) - Using Standard VNet")
            
            # ⚠️ 修复2: 完全复刻 read_data() 的预处理逻辑 (utils/__init__.py:90-93)
            # 不做transpose！保持原始(Z,Y,X)格式！
            img_dhc = np.clip(image, -75, 275)
            img_dhc = (img_dhc - img_dhc.min()) / (img_dhc.max() - img_dhc.min())
            img_dhc = img_dhc.astype(np.float32)
            
            print(f"\n[INPUT] ⭐ 复刻 read_data() 预处理 (utils/__init__.py:90-93)")
            print(f"[INPUT] Original image shape: {image.shape} [Z,Y,X]")
            print(f"[INPUT] After preprocessing: {img_dhc.shape} [保持Z,Y,X，不做transpose!]")
            print(f"[INPUT] Dtype: {img_dhc.dtype}")
            print(f"[INPUT] Value range: [{img_dhc.min():.4f}, {img_dhc.max():.4f}]")
            
            # ⚠️ 修复3: 使用 test_single_case_AB_synapse (与cloud2/test.py:227一致)
            print("\n[INFERENCE] ⭐ Running test_single_case_AB_synapse (same as cloud2/test.py:227)...")
            with torch.no_grad():
                mask, _ = test_single_case_AB_synapse(
                    model_A, model_B, img_dhc,
                    stride_xy=64, stride_z=32,
                    patch_size=config.patch_size, num_classes=config.num_cls
                )
            
            # synapse版本返回的就是正确的(Z,Y,X)格式，不需要额外transpose
            
            print(f"\n[RESULT] ⭐ Inference completed with correct logic")
            print(f"[RESULT] Mask shape: {mask.shape}")
            print(f"[RESULT] Original image shape: {orig_shape}")
            print(f"[RESULT] Shape match: {mask.shape == orig_shape}")
            print(f"[RESULT] Unique labels: {np.unique(mask)}")
            print(f"[RESULT] Label distribution:")
            unique, counts = np.unique(mask, return_counts=True)
            for label, count in zip(unique, counts):
                pct = count / mask.size * 100
                print(f"         Label {int(label):2d}: {count:8d} voxels ({pct:5.2f}%)")
        else:
            raise RuntimeError(f"DHC模型格式错误：期望{{'A': ..., 'B': ...}}，但得到keys: {list(state_dict.keys())[:5]}")
    
    # ========== SKCDF / VNet 原有逻辑（完全不动）==========
    elif 'vnet' in model_name or 'fully' in model_name:
        model = VNet(n_channels=config.num_channels, n_classes=config.num_cls,
                     n_filters=config.n_filters, normalization='batchnorm', has_dropout=False)
        
        state_dict = torch.load(model_path, map_location='cuda')
        model.load_state_dict(state_dict)
        model.cuda().eval()
        
        with torch.no_grad():
            mask, _ = test_single_case(
                model, img_norm,
                stride_xy=64, stride_z=32,
                patch_size=config.patch_size, num_classes=config.num_cls
            )
    else:
        model = VNet_Decouple_Attention_ABC(n_channels=config.num_channels, n_classes=config.num_cls,
                                            n_filters=config.n_filters, normalization='batchnorm', has_dropout=False)
        
        state_dict = torch.load(model_path, map_location='cuda')
        model.load_state_dict(state_dict)
        model.cuda().eval()
        
        with torch.no_grad():
            mask, _ = test_single_case(
                model, img_norm,
                stride_xy=64, stride_z=32,
                patch_size=config.patch_size, num_classes=config.num_cls
            )
    
    # 4. 🚨 将算好的小尺寸掩膜，强制放大回真实的物理尺寸（仅SKCDF/VNet需要）🚨
    if need_restore:
        print(f"[RESIZE] Restoring mask back to physical scale: {orig_shape}")
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0)
        mask_restored = F.interpolate(mask_tensor, size=orig_shape, mode='nearest').squeeze().numpy()
    else:
        print(f"[SKIP] DHC模型使用原始分辨率，无需尺寸恢复")
        mask_restored = mask
    
    # 5. 保存结果
    print(f"\n[SAVE] Preparing to save result...")
    print(f"[SAVE] Original image shape: {orig_shape}")
    print(f"[SAVE] Mask shape before save: {mask_restored.shape}")
    
    if need_restore:
        print(f"[RESIZE] Restoring mask back to physical scale: {orig_shape}")
        mask_tensor = torch.FloatTensor(mask_restored).unsqueeze(0).unsqueeze(0)
        mask_restored = F.interpolate(mask_tensor, size=orig_shape, mode='nearest').squeeze().numpy()
        print(f"[SAVE] After resize: {mask_restored.shape}")
    else:
        print(f"[SKIP] DHC model uses original resolution, checking shape match...")
        
        # ⭐ 关键修复：检查mask shape是否与原图匹配
        if mask_restored.shape != orig_shape:
            print(f"[WARN] Shape mismatch! Mask: {mask_restored.shape} vs Original: {orig_shape}")
            print(f"[WARN] Attempting to adjust mask dimensions...")
            
            try:
                mask_tensor = torch.FloatTensor(mask_restored).unsqueeze(0).unsqueeze(0)
                mask_restored = F.interpolate(mask_tensor, size=orig_shape, mode='nearest').squeeze().numpy()
                print(f"[SAVE] After interpolation: {mask_restored.shape}")
            except Exception as interp_err:
                print(f"[WARN] Interpolation failed: {str(interp_err)[:100]}")
                print(f"[WARN] Trying transpose adjustments...")
                
                # 尝试常见的转置组合
                if len(orig_shape) == 3 and len(mask_restored.shape) == 3:
                    if mask_restored.shape == (orig_shape[2], orig_shape[1], orig_shape[0]):
                        mask_restored = mask_restored.transpose(2, 1, 0)
                        print(f"[SAVE] After transpose(2,1,0): {mask_restored.shape}")
                    elif mask_restored.shape == (orig_shape[1], orig_shape[2], orig_shape[0]):
                        mask_restored = mask_restored.transpose(2, 0, 1)
                        print(f"[SAVE] After transpose(2,0,1): {mask_restored.shape}")
                    elif mask_restored.shape == (orig_shape[0], orig_shape[2], orig_shape[1]):
                        mask_restored = mask_restored.transpose(0, 2, 1)
                        print(f"[SAVE] After transpose(0,2,1): {mask_restored.shape}")
        
        print(f"[SAVE] Final mask shape: {mask_restored.shape} (target: {orig_shape})")
    
    out_itk = sitk.GetImageFromArray(mask_restored.astype(np.uint8))
    
    try:
        out_itk.CopyInformation(itk_img)
        print("[SAVE] ✓ Copied spatial metadata from original image")
    except Exception as meta_err:
        print(f"[WARN] Could not copy metadata: {str(meta_err)[:100]}")
        print("[WARN] Saving without original metadata (using default)")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '/tmp', exist_ok=True)
    sitk.WriteImage(out_itk, output_path)
    
    print(f"\n[DONE] ✓ Result saved to: {output_path}")
    print("=" * 60)
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, default='synapse', choices=['synapse', 'amos'])
    args = parser.parse_args()
    
    infer_with_test_logic(args.input, args.output, args.model, args.task)