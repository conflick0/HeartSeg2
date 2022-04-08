import os
from os import path
import shutil


def cp_files(inp_dir, dst_dir,  is_mask=False):
    '''重新命名醫院CT資料'''
    if is_mask:
        fns = sorted(os.listdir(inp_dir), key=lambda x: int(x.split('.')[0]))
    else:
        fns = os.listdir(inp_dir)

    for i, inp_fn in enumerate(fns):
        shutil.copyfile(path.join(inp_dir, inp_fn), path.join(dst_dir, f'ID0_{i}.jpg'))


if __name__ == '__main__':
    inp_base_dir = 'D:\dataset\corcta'
    out_base_dir = path.join(inp_base_dir, 'corcta_data')
    inp_dirs = ['corcta_dcm_jpg', 'corcta_adj_contract_jpg', 'mask_jpg']
    out_dirs = ['corcta', 'corcta_adj_contract', 'mask']
    os.makedirs(out_base_dir, exist_ok=True)
    for inp_dir, out_dir in zip(inp_dirs, out_dirs):
        inp_dir = path.join(inp_base_dir, inp_dir)
        out_dir = path.join(out_base_dir, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        cp_files(inp_dir, out_dir)
