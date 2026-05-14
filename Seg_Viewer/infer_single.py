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
from utils import test_single_case, test_single_case_AB_synapse, Config


def infer_with_test_logic(input_path, output_path, model_path, task='synapse'):
    print("=" * 60)
    print("[START] Unified Inference")
    print("=" * 60)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = Config(task)

    state_dict = torch.load(model_path, map_location='cuda', weights_only=False)
    is_dhc = 'A' in state_dict and 'B' in state_dict

    itk_img = sitk.ReadImage(input_path)
    image_sitk = sitk.GetArrayFromImage(itk_img).astype(np.float32)
    original_shape = image_sitk.shape
    print(f"\n[IMAGE] Original (SITK Z,Y,X): {original_shape}")

    target_shape = (
        config.patch_size[0] + config.patch_size[0] // 4,
        config.patch_size[1] + config.patch_size[1] // 4,
        config.patch_size[2] + config.patch_size[2] // 4,
    )
    print(f"[TARGET] Training scale: {target_shape}")

    if image_sitk.shape != target_shape:
        print(f"[RESIZE] {image_sitk.shape} -> {target_shape}")
        img_tensor = torch.FloatTensor(image_sitk).unsqueeze(0).unsqueeze(0)
        img_resized = F.interpolate(img_tensor, size=target_shape, mode='trilinear', align_corners=False)
        image = img_resized.squeeze().numpy()
    else:
        image = image_sitk

    image = image.clip(min=-75, max=275)
    image = (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.float32)
    print(f"[NORMALIZE] [{image.min():.4f}, {image.max():.4f}], shape: {image.shape}")

    if is_dhc:
        print("\n[MODEL] DHC dual-model")
        model_A = VNet(n_channels=config.num_channels, n_classes=config.num_cls,
                       n_filters=config.n_filters, normalization='batchnorm', has_dropout=False)
        model_B = VNet(n_channels=config.num_channels, n_classes=config.num_cls,
                       n_filters=config.n_filters, normalization='batchnorm', has_dropout=False)
        model_A.load_state_dict(state_dict['A'])
        model_B.load_state_dict(state_dict['B'])
        model_A.cuda().eval()
        model_B.cuda().eval()

        with torch.no_grad():
            mask, _ = test_single_case_AB_synapse(
                model_A, model_B, image,
                stride_xy=32, stride_z=16,
                patch_size=config.patch_size, num_classes=config.num_cls
            )
    else:
        print("\n[MODEL] SKCDF single-model")
        model = VNet_Decouple_Attention_ABC(
            n_channels=config.num_channels, n_classes=config.num_cls,
            n_filters=config.n_filters, normalization='batchnorm', has_dropout=False
        )
        model.load_state_dict(state_dict)
        model.cuda().eval()

        with torch.no_grad():
            mask, _ = test_single_case(
                model, image,
                stride_xy=64, stride_z=32,
                patch_size=config.patch_size, num_classes=config.num_cls
            )

    print(f"[RESULT] Mask: {mask.shape}, labels: {np.unique(mask)}")

    if mask.shape != original_shape:
        print(f"\n[RESTORE] {mask.shape} -> {original_shape}")
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0)
        mask_restored = F.interpolate(mask_tensor, size=original_shape, mode='nearest').squeeze().numpy().astype(np.int16)
    else:
        mask_restored = mask.astype(np.int16)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '/tmp', exist_ok=True)
    out_itk = sitk.GetImageFromArray(mask_restored.astype(np.float32))
    out_itk.CopyInformation(itk_img)
    sitk.WriteImage(out_itk, output_path)

    print(f"\n[DONE] Saved to: {output_path} (shape: {mask_restored.shape})")
    print("=" * 60)
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified inference')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, default='synapse', choices=['synapse', 'amos'])
    args = parser.parse_args()
    infer_with_test_logic(args.input, args.output, args.model, args.task)
