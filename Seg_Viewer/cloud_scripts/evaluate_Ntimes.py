import os
import sys
import argparse
import numpy as np
import nibabel as nib
from medpy import metric

sys.path.insert(0, '/workspace')
from utils.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse', choices=['synapse', 'amos'])
parser.add_argument('--pred', type=str, required=True, help='预测结果 NIfTI 文件')
parser.add_argument('--label', type=str, required=True, help='Ground Truth NIfTI 文件')
args = parser.parse_args()

config = Config(args.task)

if __name__ == '__main__':
    print(f"[EVAL] Starting evaluation for task: {args.task.upper()}")
    print(f"[PRED] {args.pred}")
    print(f"[LABEL] {args.label}")
    
    # 1. 加载预测与真实标签 (NIfTI 格式)
    pred = nib.load(args.pred).get_fdata(dtype=np.float32).astype(np.int8)
    label = nib.load(args.label).get_fdata(dtype=np.float32).astype(np.int8)
    
    print(f"[PRED] Initial Shape: {pred.shape}, Unique: {np.unique(pred)}")
    print(f"[LABEL] Initial Shape: {label.shape}, Unique: {np.unique(label)}")
    
    # 2. 维度对齐保护 (兼容预处理数据与原始NIfTI)
    if pred.shape != label.shape:
        print(f"[ALIGN] Shape mismatch! Aligning pred to label...")
        if pred.shape == (label.shape[2], label.shape[0], label.shape[1]):
            pred = pred.transpose(1, 2, 0)
        elif pred.shape == (label.shape[1], label.shape[2], label.shape[0]):
            pred = pred.transpose(2, 0, 1)
        elif pred.shape == (label.shape[2], label.shape[1], label.shape[0]):
            pred = pred.transpose(2, 1, 0)
        print(f"[PRED] Aligned Shape: {pred.shape}")
    else:
        print(f"[ALIGN] Shapes match perfectly.")

    # 3. 动态配置验证类别
    test_cls = [i for i in range(1, config.num_cls)]
    
    results = {}
    for i in test_cls:
        pred_i = (pred == i)
        label_i = (label == i)
        
        if pred_i.sum() > 0 and label_i.sum() > 0:
            dice = metric.binary.dc(pred_i, label_i) * 100
            try:
                # OOM 保护：如果云端算 ASD 也爆内存，则捕获跳过
                asd = metric.binary.asd(pred_i, label_i)
            except MemoryError:
                print(f"[WARNING] Memory error calculating ASD for class {i}, returning NaN")
                asd = np.nan
                
            results[i] = {'dice': dice, 'asd': asd}
        elif pred_i.sum() > 0 and label_i.sum() == 0:
            results[i] = {'dice': 0.0, 'asd': 128.0}
        elif pred_i.sum() == 0 and label_i.sum() > 0:
            results[i] = {'dice': 0.0, 'asd': 128.0}
        else:
            results[i] = {'dice': 100.0, 'asd': 0.0}
    
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({args.task.upper()})")
    print("="*60)
    
    # 4. 动态器官字典支持
    organ_names_synapse = {
        1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gallbladder',
        5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'ivc',
        9: 'portal_vein', 10: 'pancreas', 11: 'adrenal_gland_right',
        12: 'adrenal_gland_left', 13: 'esophagus'
    }
    
    organ_names_amos = {
        1: 'spleen', 2: 'right_kidney', 3: 'left_kidney', 4: 'gallbladder',
        5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta',
        9: 'ivc', 10: 'pancreas', 11: 'adrenal_gland_right',
        12: 'adrenal_gland_left', 13: 'duodenum', 14: 'bladder', 15: 'prostate_uterus'
    }
    
    organ_names = organ_names_amos if args.task.lower() == 'amos' else organ_names_synapse
    
    dice_list = []
    asd_list = []
    
    for cls_id in test_cls:
        name = organ_names.get(cls_id, f'Class_{cls_id}')
        d = results[cls_id]['dice']
        a = results[cls_id]['asd']
        dice_list.append(d)
        
        if not np.isnan(a):
            asd_list.append(a)
            print(f"{name:20s}: DICE={d:6.2f}%   ASD={a:6.2f}")
        else:
            print(f"{name:20s}: DICE={d:6.2f}%   ASD=NaN (OOM)")
    
    print("="*60)
    avg_dice = np.mean(dice_list)
    avg_asd = np.mean(asd_list) if len(asd_list) > 0 else np.nan
    print(f"{'AVERAGE':20s}: DICE={avg_dice:6.2f}%   ASD={avg_asd:6.2f}")
    print("="*60)