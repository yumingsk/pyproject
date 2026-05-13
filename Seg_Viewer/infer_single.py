import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, '/workspace')

from models.vnet import VNet
from models.SKCDF import VNet_Decouple_Attention_ABC
from utils import test_single_case, read_data, Config

def infer_with_test_logic(input_path, output_path, model_path, task='synapse'):
    """
    完全使用 test.py 的逻辑和参数配置
    通过 Config 类获取正确的模型参数
    """
    print("=" * 60)
    print("[START] Single file inference (Using test.py logic)")
    print(f"[INPUT] {input_path}")
    print(f"[MODEL] {model_path}")
    print(f"[OUTPUT] {output_path}")
    print(f"[TASK] {task}")
    print("=" * 60)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = Config(task)
    
    print(f"\n[CONFIG] Using Config from test.py:")
    print(f"  - num_channels: {config.num_channels}")  
    print(f"  - num_cls: {config.num_cls}")
    print(f"  - n_filters: {config.n_filters}")
    print(f"  - patch_size: {config.patch_size}")
    
    image = nib.load(input_path).get_fdata(dtype=np.float32)
    original_shape = image.shape
    print(f"\n[IMAGE] Shape: {original_shape}, Range: [{image.min():.2f}, {image.max():.2f}]")
    
    model_name = os.path.basename(model_path).lower()
    
    if 'vnet' in model_name or 'fully' in model_name:
        print(f"\n[MODEL] Loading VNet (like test.py): {model_path}")
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        )
    else:
        print(f"\n[MODEL] Loading SKCDF (like test.py): {model_path}")
        model = VNet_Decouple_Attention_ABC(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        )
    
    state_dict = torch.load(model_path, map_location='cuda')
    model.load_state_dict(state_dict)
    model.cuda().eval()
    print("[MODEL] ✓ Model loaded successfully")
    
    img_trans = image.transpose(2, 0, 1) 
    img_norm = np.clip(img_trans, -125, 275)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    
    print(f"\n[TENSOR] Shape: {img_norm.shape} (original resolution)")
    print(f"[STRIDE] xy=64, z=32 (speed=1 like test.py)")
    print(f"[PATCH] size={config.patch_size}")
    
    print("\n[INFERENCE] Running...")
    with torch.no_grad():
        mask, _ = test_single_case(
            model,
            img_norm,
            stride_xy=64,
            stride_z=32,
            patch_size=config.patch_size,
            num_classes=config.num_cls
        )
    
    print(f"[RESULT] Mask shape: {mask.shape}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '/tmp', exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.int16), np.eye(4)), output_path)
    
    print(f"\n[DONE] ✓ Result saved to: {output_path}")
    print(f"[RESULT] Unique labels: {np.unique(mask)}")
    print("=" * 60)
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single CT inference (test.py logic)')
    parser.add_argument('--input', type=str, required=True, help='Input NIfTI file path')
    parser.add_argument('--output', type=str, required=True, help='Output NIfTI file path')
    parser.add_argument('--model', type=str, required=True, help='Model .pth path')
    parser.add_argument('--task', type=str, default='synapse', choices=['synapse', 'amos'], help='Dataset type')
    
    args = parser.parse_args()
    
    infer_with_test_logic(args.input, args.output, args.model, args.task)