#!/usr/bin/env python3
"""
使用 yudaun/test.py 的逻辑进行单文件推理
"""

import os, sys, argparse
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, '/workspace')

from models.vnet import VNet
from models.SKCDF import VNet_Decouple_Attention_ABC
from utils import test_single_case, Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', default='synapse')
    parser.add_argument('--speed', type=int, default=1)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = Config(args.task)  # ← 关键！用Config获取参数
    
    stride_dict = {0: (32, 16), 1: (64, 16), 2: (128, 32)}
    stride = stride_dict[args.speed]
    
    model_name = os.path.basename(args.model).lower()
    
    if 'vnet' in model_name or 'fully' in model_name:
        model = VNet(
            n_channels=config.num_channels,  # = 1 ✓
            n_classes=config.num_cls,        # = 14 ✓
            n_filters=config.n_filters,      # = 32 ✓
            normalization='batchnorm',
            has_dropout=False
        )
    else:
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
    
    image = nib.load(args.input).get_fdata(dtype=np.float32)
    img_trans = image.transpose(2, 0, 1)
    img_norm = np.clip(img_trans, -125, 275)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    
    with torch.no_grad():
        mask, _ = test_single_case(
            model, img_norm,
            stride_xy=stride[0],
            stride_z=stride[1],
            patch_size=config.patch_size,
            num_classes=config.num_cls
        )
    
    nib.save(nib.Nifti1Image(mask.astype(np.int16), np.eye(4)), args.output)
    print(f"[DONE] Saved to {args.output}")

if __name__ == '__main__':
    main()