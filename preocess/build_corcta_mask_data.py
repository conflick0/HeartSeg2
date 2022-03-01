import os

import numpy as np
import nrrd
from PIL import Image
from os import path
from predict.utils.img_loader import write_pred_mask


def nrrd2jpg(file_path, out_dir):
    '''讀入 nrrd 遮罩，輸出 jpg 遮罩'''
    os.makedirs(out_dir, exist_ok=True)
    data, header = nrrd.read(file_path, index_order='C')
    data = np.flip(data, 0)
    for i, d in enumerate(data):
        img = Image.fromarray(d * 255.).convert('RGB')
        img.save(path.join(out_dir, f'{i}.jpg'))


def draw_mask(inp_dir, msk_dir, out_dir):
    '''將遮罩畫到原始影像上'''
    os.makedirs(out_dir, exist_ok=True)
    inp_fns = os.listdir(inp_dir)
    msk_fns = sorted(os.listdir(msk_dir), key=lambda x: int(x.split('.')[0]))
    for i, (inp_fn, msk_fn) in enumerate(zip(inp_fns, msk_fns)):
        inp_img = Image.open(path.join(inp_dir, inp_fn)).convert("L")
        msk_img = Image.open(path.join(msk_dir, msk_fn)).convert("L")
        write_pred_mask(inp_img, msk_img, path.join(out_dir, f'{i}.bmp'))


def rename_inp_file_name(inp_dir):
    '''重新命名醫院CT資料'''
    for i, inp_fn in enumerate(os.listdir(inp_dir)):
        os.rename(path.join(inp_dir, inp_fn), path.join(inp_dir, f'ID0_{i}.jpg'))


if __name__ == '__main__':
    nrrd2jpg(
        file_path=r'D:\dataset\corcta\corcta_project\v2\export_data\Segmentation.nrrd',
        out_dir='D:\dataset\corcta\mask_jpg'
    )

    draw_mask(
        inp_dir='D:\dataset\corcta\corcta_dcm_jpg',
        msk_dir='D:\dataset\corcta\mask_jpg',
        out_dir='D:\dataset\corcta\cmp',
    )

    # rename_inp_file_name(r'D:\dataset\corcta\tr_corcta\masks')
