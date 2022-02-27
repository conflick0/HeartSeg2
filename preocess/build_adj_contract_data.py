import os

import numpy as np
from skimage import exposure
from tqdm import tqdm
from PIL import Image

from preocess.dataset import Dataset, load_img


def adj_contrast_img(img):
    '''對比度拉伸，來調整對比度'''
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def build_chest_ct_data(src_dir, dst_dir):
    '''建立 chest ct data'''
    root_src_img_dir = os.path.join(src_dir, 'images')
    root_src_lab_dir = os.path.join(src_dir, 'masks')

    root_dst_img_dir = os.path.join(dst_dir, 'images')
    root_dst_lab_dir = os.path.join(dst_dir, 'masks')

    os.makedirs(root_dst_img_dir, exist_ok=True)
    os.makedirs(root_dst_lab_dir, exist_ok=True)

    pids = sorted(os.listdir(root_src_img_dir))

    for pid in tqdm(pids):
        src_img_dir = os.path.join(root_src_img_dir, pid)
        src_lab_dir = os.path.join(root_src_lab_dir, pid)

        dst_img_dir = os.path.join(root_dst_img_dir, pid)
        dst_lab_dir = os.path.join(root_dst_lab_dir, pid)

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lab_dir, exist_ok=True)

        x_fs = sorted(os.listdir(src_img_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        y_fs = sorted(os.listdir(src_lab_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        x_pths = [os.path.join(src_img_dir, x_f) for x_f in x_fs]
        y_pths = [os.path.join(src_lab_dir, y_f) for y_f in y_fs]

        ds = Dataset(x_pths, y_pths, is_rm_artifacts=False)

        for i, (x, y) in enumerate(ds):
            x = adj_contrast_img(x)

            x_im = Image.fromarray(x).convert("L")
            y_im = Image.fromarray(y).convert("L")

            x_im.save(os.path.join(dst_img_dir, f'{pid}_{i}.jpg'))
            y_im.save(os.path.join(dst_lab_dir, f'{pid}_mask_{i}.jpg'))

def build_corcta_data(src_dir, dst_dir):
    '''建立醫院提供的資料'''
    os.makedirs(dst_dir, exist_ok=True)
    img_ns = sorted(os.listdir(src_dir))
    for img_n in img_ns:
        img = load_img(os.path.join(src_dir, img_n))
        img = adj_contrast_img(img)
        img = Image.fromarray(img).convert("L")
        img.save(os.path.join(dst_dir, img_n))


if __name__ == '__main__':
    '''
    將資料集進行影像對比度拉伸處理，使訓練資料與測試資料影像對比度盡量相似。
    '''
    is_build_chest_ct = False
    if is_build_chest_ct:
        build_chest_ct_data(
            src_dir=r'D:\home\school\ntut\dataset\chest-ct-segmentation\crop2_data',
            dst_dir=r'D:\home\school\ntut\dataset\chest-ct-segmentation\adj_contract_data'
        )
    else:
        build_corcta_data(
            src_dir=r'D:\home\school\ntut\dataset\corcta\corcta_dcm_jpg',
            dst_dir=r'D:\home\school\ntut\dataset\corcta\corcta_adj_contract_jpg'
        )