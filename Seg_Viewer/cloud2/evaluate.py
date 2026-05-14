import os
import numpy as np
import argparse
from medpy import metric
from tqdm import tqdm
from utils import read_list, read_nifti
from utils.config import Config # 修改点 1
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="Task_synapse_20p/dhcnew_synapse_20p")
    args = parser.parse_args()

    config = Config("synapse") # 修改点 2：实例化
    test_cls = [i for i in range(1, config.num_cls)]
    values = np.zeros((len(test_cls), 2)) 
    ids_list = read_list('test')
    
    for data_id in tqdm(ids_list):
        # 修改点 3：对齐你的 fold1/predictions_AB 路径
        pred_path = os.path.join("./logs", args.exp, "fold1/predictions_AB", f'{data_id}.nii.gz')
        pred = read_nifti(pred_path)
        
        # 修改点 4：确保 label 路径正确（通常在 npy 同级或 dataset 目录下）
        label_path = os.path.join(config.base_dir, 'labelsTr', f'label{data_id}.nii.gz')
        label = read_nifti(label_path)

        # 修改点 5：删掉 label 的 F.interpolate 缩放逻辑！！！
        
        for i in test_cls:
            pred_i = (pred == i)
            label_i = (label == i)
            if pred_i.sum() > 0 and label_i.sum() > 0:
                dice = metric.binary.dc(pred_i, label_i) * 100
                hd95 = metric.binary.hd95(pred_i, label_i)
                values[i - 1] += np.array([dice, hd95])

    values /= len(ids_list)
    print("====== Dice ======")
    print(np.round(values[:,0],1))
    print("====== HD ======")
    print(np.round(values[:,1],1))
    print(np.mean(values, axis=0)[0], np.mean(values, axis=0)[1])
