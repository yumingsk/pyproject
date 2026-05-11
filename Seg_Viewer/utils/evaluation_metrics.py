import numpy as np
import pandas as pd
from medpy import metric

ORGAN_ABBREVIATIONS = {
    '脾脏': 'Spl',
    '右肾': 'R-Kid',
    '左肾': 'L-Kid',
    '胆囊': 'Gal',
    '食道': 'Eso',
    '肝脏': 'Liv',
    '胃': 'Sto',
    '主动脉': 'Aor',
    '下腔静脉': 'IVC',
    '门静脉/脾静脉': 'Veins',
    '胰腺': 'Pan',
    '右肾上腺': 'R-Adr',
    '左肾上腺': 'L-Adr',
    '十二指肠': 'Duo',
    '膀胱': 'Bla',
    '前列腺': 'Pros',
    '子宫': 'Ute'
}


def calculate_dice_asd(pred, gt, num_classes, label_names=None):
    results = {}
    
    for cls in range(1, num_classes):
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)
        
        class_name = label_names.get(cls, f"类别{cls}") if label_names else f"类别{cls}"
        class_abbr = ORGAN_ABBREVIATIONS.get(class_name, class_name[:3])
        
        if pred_mask.sum() == 0 and gt_mask.sum() == 0:
            dice = 100.0
            asd = 0.0
        elif pred_mask.sum() == 0 or gt_mask.sum() == 0:
            dice = 0.0
            asd = 128.0
        else:
            dice = metric.binary.dc(pred_mask.astype(np.uint8), gt_mask.astype(np.uint8)) * 100
            
            try:
                asd = metric.binary.asd(pred_mask.astype(np.uint8), gt_mask.astype(np.uint8))
            except MemoryError:
                asd = -1.0
        
        results[class_abbr] = {
            'DICE (%)': round(float(dice), 2),
            'ASD (mm)': round(float(asd), 2) if asd >= 0 else 'N/A',
            'full_name': class_name
        }
    
    return results


def evaluate_multiple_models_with_highlight(predictions, gt, num_classes, label_names=None, model_names=None):
    all_results = {}
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        if model_names and i < len(model_names):
            display_name = model_names[i]
        else:
            display_name = model_name
        
        results = calculate_dice_asd(pred, gt, num_classes, label_names)
        all_results[display_name] = results
    
    class_names = list(list(all_results.values())[0].keys())
    model_names_list = list(all_results.keys())
    
    best_dice_per_class = {}
    best_asd_per_class = {}
    
    for cls_name in class_names:
        dice_values = {}
        asd_values = {}
        for model_name, results in all_results.items():
            if cls_name in results:
                dice_values[model_name] = results[cls_name]['DICE (%)']
                asd_val = results[cls_name]['ASD (mm)']
                if isinstance(asd_val, (int, float)):
                    asd_values[model_name] = asd_val
        
        if dice_values:
            best_dice_model = max(dice_values, key=dice_values.get)
            best_dice_per_class[cls_name] = best_dice_model
        
        if asd_values:
            best_asd_model = min(asd_values, key=asd_values.get)
            best_asd_per_class[cls_name] = best_asd_model
    
    dice_rows = []
    asd_rows = []
    
    for cls_name in class_names:
        dice_row = {'类别': cls_name}
        asd_row = {'类别': cls_name}
        
        for model_name in model_names_list:
            if cls_name in all_results[model_name]:
                dice_val = all_results[model_name][cls_name]['DICE (%)']
                asd_val = all_results[model_name][cls_name]['ASD (mm)']
                
                if best_dice_per_class.get(cls_name) == model_name:
                    dice_row[model_name] = f"**{dice_val:.2f}**"
                else:
                    dice_row[model_name] = f"{dice_val:.2f}"
                
                if isinstance(asd_val, (int, float)):
                    if best_asd_per_class.get(cls_name) == model_name:
                        asd_row[model_name] = f"**{asd_val:.2f}**"
                    else:
                        asd_row[model_name] = f"{asd_val:.2f}"
                else:
                    asd_row[model_name] = str(asd_val)
            else:
                dice_row[model_name] = "N/A"
                asd_row[model_name] = "N/A"
        
        dice_rows.append(dice_row)
        asd_rows.append(asd_row)
    
    mean_dice_row = {'类别': '平均值'}
    mean_asd_row = {'类别': '平均值'}
    
    for model_name in model_names_list:
        dice_values = [all_results[model_name][cls]['DICE (%)'] for cls in class_names]
        asd_values = [all_results[model_name][cls]['ASD (mm)'] for cls in class_names if isinstance(all_results[model_name][cls]['ASD (mm)'], (int, float))]
        
        mean_dice = np.mean(dice_values)
        mean_asd = np.mean(asd_values) if asd_values else 0
        
        mean_dice_row[model_name] = f"{mean_dice:.2f}"
        mean_asd_row[model_name] = f"{mean_asd:.2f}"
    
    dice_rows.append(mean_dice_row)
    asd_rows.append(mean_asd_row)
    
    df_dice = pd.DataFrame(dice_rows)
    df_asd = pd.DataFrame(asd_rows)
    
    organ_mapping = {}
    for cls_name in class_names:
        if cls_name in list(all_results.values())[0]:
            organ_mapping[cls_name] = list(all_results.values())[0][cls_name].get('full_name', cls_name)
    
    return {
        'dice_df': df_dice,
        'asd_df': df_asd,
        'model_names': model_names_list,
        'class_names': class_names,
        'organ_mapping': organ_mapping
    }


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
