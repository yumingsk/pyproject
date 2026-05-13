#!/usr/bin/env python3
"""
run_single_infer.py - 单文件推理脚本
使用 yudaun/test.py 的逻辑，支持直接指定模型路径
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, '/workspace')

from models.vnet import VNet
from models.SKCDF import VNet_Decouple_Attention_ABC
from utils import test_single_case, Config

def main():
    parser = argparse.ArgumentParser(description='单文件推理 (基于test.py逻辑)')
    parser.add_argument('--input', required=True, help='输入 NIfTI 文件')
    parser.add_argument('--output', required=True, help='输出 NIfTI 文件')
    parser.add_argument('--model', required=True, help='模型 .pth 路径')
    parser.add_argument('--task', default='synapse', choices=['synapse', 'amos'])
    parser.add_argument('--speed', type=int, default=1, choices=[0, 1, 2])
    
    args = parser.parse_args()
    
    print(f"[START] Single file inference")
    print(f"[INPUT] {args.input}")
    print(f"[MODEL] {args.model}")
    print(f"[OUTPUT] {args.output}")
    print(f"[TASK] {args.task}")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = Config(args.task)
    
    stride_dict = {
        0: (32, 16),
        1: (64, 16),
        2: (128, 32),
    }
    stride = stride_dict[args.speed]
    
    model_name = os.path.basename(args.model).lower()
    
    if 'vnet' in model_name or 'fully' in model_name:
        print(f"[MODEL] Loading VNet")
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        )
    else:
        print(f"[MODEL] Loading SKCDF")
        model = VNet_Decouple_Attention_ABC(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        )
    
    model.cuda().eval()
    
    state_dict = torch.load(args.model, map_location='cuda')
    model.load_state_dict(state_dict)
    print("[MODEL] ✓ Loaded successfully")
    
    image = nib.load(args.input).get_fdata(dtype=np.float32)
    print(f"[IMAGE] Shape: {image.shape}")
    
    img_trans = image.transpose(2, 0, 1)
    img_norm = np.clip(img_trans, -125, 275)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    
    print(f"[INFERENCE] Running with stride={stride}, patch_size={config.patch_size}")
    
    with torch.no_grad():
        mask, _ = test_single_case(
            model,
            img_norm,
            stride_xy=stride[0],
            stride_z=stride[1],
            patch_size=config.patch_size,
            num_classes=config.num_cls
        )
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.int16), np.eye(4)), args.output)
    
    print(f"[DONE] ✓ Saved to {args.output}")
    print(f"[RESULT] Labels: {np.unique(mask)}")

if __name__ == '__main__':
    main()