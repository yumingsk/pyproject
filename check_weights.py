import torch

state_dict = torch.load(r'd:\PythonProject2\Seg_Viewer\ckpts\amos\DHC_amos5p.pth', map_location='cpu', weights_only=False)
keys = list(state_dict['A'].keys())

print('权重文件中的键名（前30个）:')
print('\n'.join(keys[:30]))
print(f'\n总共有 {len(keys)} 个键')

print('\n查找decoder相关的键:')
decoder_keys = [k for k in keys if 'decoder' in k]
print(f'找到 {len(decoder_keys)} 个decoder相关的键')
if decoder_keys:
    print('\n'.join(decoder_keys[:10]))
else:
    print('没有找到decoder相关的键')
    
print('\n查找block_five_up相关的键:')
block_five_keys = [k for k in keys if 'block_five_up' in k]
print(f'找到 {len(block_five_keys)} 个block_five_up相关的键')
if block_five_keys:
    print('\n'.join(block_five_keys[:10]))
