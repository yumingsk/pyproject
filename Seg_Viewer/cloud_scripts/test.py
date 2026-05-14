import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, '/workspace')

from models.vnet import VNet
from models.SKCDF import VNet_Decouple_Attention_ABC
from utils import test_single_case, Config, maybe_mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--input', type=str, required=True, help='输入 NIfTI 文件路径')
parser.add_argument('--output', type=str, required=True, help='输出 NIfTI 文件路径')
parser.add_argument('--model', type=str, required=True, help='模型 .pth 文件路径')
parser.add_argument('--speed', type=int, default=1)
parser.add_argument('-g', '--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(args.task)

stride_dict = {
    0: (32, 16),
    1: (64, 16),
    2: (128, 32),
}
stride = stride_dict[args.speed]

if __name__ == '__main__':
    maybe_mkdir(os.path.dirname(args.output) if os.path.dirname(args.output) else '/tmp')
    
    model_name = os.path.basename(args.model).lower()
    
    if 'vnet' in model_name or 'fully' in model_name:
        print(f'[MODEL] Loading VNet: {args.model}')
        model = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model.eval()
    else:
        print(f'[MODEL] Loading SKCDF: {args.model}')
        model = VNet_Decouple_Attention_ABC(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).cuda()
        model.eval()

    with torch.no_grad():
        model.load_state_dict(torch.load(args.model, map_location='cuda'))
        print(f'[MODEL] ✓ Loaded from {args.model}')
        
        image = nib.load(args.input).get_fdata(dtype=np.float32)
        print(f'[IMAGE] Shape: {image.shape}')
        
        img_trans = image.transpose(2, 0, 1)
        img_norm = np.clip(img_trans, -125, 275)
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
        
        print(f'[INFERENCE] Running with stride={stride}, patch_size={config.patch_size}')
        
        mask, _ = test_single_case(
            model,
            img_norm,
            stride_xy=stride[0],
            stride_z=stride[1],
            patch_size=config.patch_size,
            num_classes=config.num_cls
        )
        
        nib.save(nib.Nifti1Image(mask.astype(np.int16), np.eye(4)), args.output)
        
        print(f'[DONE] ✓ Prediction saved to {args.output}')
        print(f'[RESULT] Unique labels: {np.unique(mask)}')