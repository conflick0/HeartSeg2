import os
from PIL import Image
from tqdm import tqdm
import csv

from img_loader import list_sorted_test_dir, load_dcm


def crop_img(img_path, box):
    img = Image.open(img_path)
    img = img.crop(box)
    return img


def crop_img_by_pid(pid, box):
    src_p_inp_dir = os.path.join(src_inp_dir, pid)
    src_p_lab_dir = os.path.join(src_lab_dir, pid)

    dst_p_inp_dir = os.path.join(dst_inp_dir, pid)
    dst_p_lab_dir = os.path.join(dst_lab_dir, pid)

    for d in [dst_p_inp_dir, dst_p_lab_dir]:
        os.makedirs(os.path.join(d), exist_ok=True)

    x_fs = list_sorted_test_dir(src_p_inp_dir)
    y_fs = list_sorted_test_dir(src_p_lab_dir)

    for x_f, y_f in zip(x_fs, y_fs):
        x_path = os.path.join(src_p_inp_dir, x_f)
        y_path = os.path.join(src_p_lab_dir, y_f)
        x = crop_img(x_path, box)
        y = crop_img(y_path, box)

        x.save(os.path.join(dst_p_inp_dir, x_f))
        y.save(os.path.join(dst_p_lab_dir, y_f))


def crop():
    pids = sorted(os.listdir(src_inp_dir))

    with open(r'D:\home\school\ntut\lab\project\HeartSeg\data_csv\rm_pid.csv', newline='', encoding='utf-8') as f:
        rows = csv.reader(f)
        for row in rows:
            pids.remove(row[0].lstrip('﻿'))

    # box: left, top, right, bottom
    box = (150, 50, 460, 340)

    for pid in tqdm(pids):
        crop_img_by_pid(pid, box)


if __name__ == '__main__':
    src_inp_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\test_data\images'
    src_lab_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\test_data\masks'

    dst_inp_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\crop_data\images'
    dst_lab_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\crop_data\masks'

    # crop img
    # crop()




