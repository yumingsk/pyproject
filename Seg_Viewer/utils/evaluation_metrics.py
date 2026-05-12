import numpy as np
import pandas as pd
from medpy import metric


def calculate_dice_asd(pred, gt, num_classes, label_names=None):
    results = {}
    
    for cls in range(1, num_classes):
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)
        
        class_name = label_names.get(cls, f"类别{cls}") if label_names else f"类别{cls}"
        
        if pred_mask.sum() == 0 and gt_mask.sum() == 0:
            dice = 100.0
            asd = 0.0
        elif pred_mask.sum() == 0 or gt_mask.sum() == 0:
            dice = 0.0
            asd = 128.0
        else:
            dice = metric.binary.dc(pred_mask.astype(np.uint8), gt_mask.astype(np.uint8)) * 100
            asd = metric.binary.asd(pred_mask.astype(np.uint8), gt_mask.astype(np.uint8))
        
        results[class_name] = {
            'DICE (%)': round(dice, 2),
            'ASD (mm)': round(asd, 2)
        }
    
    return results


def evaluate_single_case(pred, gt, num_classes, label_names=None, case_name="Case"):
    results = calculate_dice_asd(pred, gt, num_classes, label_names)
    
    df = pd.DataFrame(results).T
    df.columns = ['DICE (%)', 'ASD (mm)']
    df.index.name = case_name
    
    mean_dice = df['DICE (%)'].mean()
    mean_asd = df['ASD (mm)'].mean()
    
    df.loc['平均值'] = [round(mean_dice, 2), round(mean_asd, 2)]
    
    return df


def evaluate_multiple_models(predictions, gt, num_classes, label_names=None, model_names=None):
    all_results = {}
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        if model_names and i < len(model_names):
            display_name = model_names[i]
        else:
            display_name = model_name
        
        results = calculate_dice_asd(pred, gt, num_classes, label_names)
        all_results[display_name] = results
    
    table_data = []
    class_names = list(list(all_results.values())[0].keys())
    
    for cls_name in class_names:
        row = {'类别': cls_name}
        for model_name, results in all_results.items():
            if cls_name in results:
                row[f"{model_name}"] = f"D:{results[cls_name]['DICE (%)']} | A:{results[cls_name]['ASD (mm)']}"
            else:
                row[f"{model_name}"] = "N/A"
        table_data.append(row)
    
    mean_row = {'类别': '平均值'}
    for model_name, results in all_results.items():
        dice_values = [r['DICE (%)'] for r in results.values()]
        asd_values = [r['ASD (mm)'] for r in results.values()]
        mean_dice = np.mean(dice_values)
        mean_asd = np.mean(asd_values)
        mean_row[f"{model_name}"] = f"D:{round(mean_dice, 2)} | A:{round(mean_asd, 2)}"
    table_data.append(mean_row)
    
    df = pd.DataFrame(table_data)
    
    return df


def evaluate_multiple_cases(predictions_list, gt_list, num_classes, label_names=None, case_names=None):
    all_case_results = []
    
    for i, (pred, gt) in enumerate(zip(predictions_list, gt_list)):
        case_name = case_names[i] if case_names and i < len(case_names) else f"Case_{i+1}"
        df = evaluate_single_case(pred, gt, num_classes, label_names, case_name)
        all_case_results.append(df)
    
    return all_case_results


def print_evaluation_table(df, title="评估结果"):
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    print(df.to_string())
    print('='*60)


def evaluate_multiple_models_with_highlight(predictions, gt, num_classes, label_names=None):
    all_results = {}
    
    for model_name, pred in predictions.items():
        results = calculate_dice_asd(pred, gt, num_classes, label_names)
        all_results[model_name] = results
    
    class_names = list(list(all_results.values())[0].keys())
    organ_mapping = {}
    for cls_name in class_names:
        short = ''.join([w[0] for w in cls_name.split('(')[0].split()]) if len(cls_name) > 4 else cls_name[:3]
        if len(short) < 2:
            short = cls_name[:2]
        organ_mapping[short] = cls_name
    
    dice_data = []
    asd_data = []
    for cls_name in class_names:
        dice_row = {'器官': organ_mapping.get(''.join([w[0] for w in cls_name.split('(')[0].split()]) if len(cls_name) > 4 else cls_name[:3], cls_name) if len(cls_name) > 2 else cls_name}
        asd_row = {'器官': dice_row['器官']}
        
        dice_vals = []
        asd_vals = []
        for model_name, results in all_results.items():
            d_val = results[cls_name]['DICE (%)']
            a_val = results[cls_name]['ASD (mm)']
            dice_vals.append(d_val)
            asd_vals.append(a_val)
        
        best_dice_idx = np.argmax(dice_vals)
        best_asd_idx = np.argmin(asd_vals)
        
        for i, (model_name, results) in enumerate(all_results.items()):
            d_val = results[cls_name]['DICE (%)']
            a_val = results[cls_name]['ASD (mm)']
            dice_row[model_name] = f"**{d_val:.2f}**" if i == best_dice_idx else f"{d_val:.2f}"
            asd_row[model_name] = f"**{a_val:.2f}**" if i == best_asd_idx else f"{a_val:.2f}"
        
        dice_data.append(dice_row)
        asd_data.append(asd_row)
    
    mean_dice_row = {'器官': '平均值'}
    mean_asd_row = {'器官': '平均值'}
    for model_name, results in all_results.items():
        d_mean = np.mean([r['DICE (%)'] for r in results.values()])
        a_mean = np.mean([r['ASD (mm)'] for r in results.values()])
        mean_dice_row[model_name] = f"**{d_mean:.2f}**"
        mean_asd_row[model_name] = f"**{a_mean:.2f}**"
    dice_data.append(mean_dice_row)
    asd_data.append(mean_asd_row)
    
    return {
        'dice_df': pd.DataFrame(dice_data),
        'asd_df': pd.DataFrame(asd_data),
        'organ_mapping': {v: k for k, v in organ_mapping.items()}
    }


if __name__ == "__main__":
    print("评估模块使用示例：")
    print()
    print("1. 单个模型评估：")
    print("   df = evaluate_single_case(pred, gt, num_classes, label_names)")
    print()
    print("2. 多模型对比：")
    print("   predictions = {'模型A': pred_a, '模型B': pred_b}")
    print("   df = evaluate_multiple_models(predictions, gt, num_classes)")
    print()
    print("3. 多样本评估：")
    print("   dfs = evaluate_multiple_cases([pred1, pred2], [gt1, gt2], num_classes)")
    print()
    print("4. 打印表格：")
    print("   print_evaluation_table(df, '评估结果')")
