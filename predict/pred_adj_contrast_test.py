import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from predict.datasets.test_dataset import TestDataset
from predict.utils.img_loader import rescale_img, threshold_img, write_pred_mask, write_pred_label_mask
from predict.nets.net import Net


def mk_predict_dir(dir):
    os.makedirs(os.path.join(dir), exist_ok=True)
    os.makedirs(os.path.join(dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'segs'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'mask_results'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'gts'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'gt_results'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'comparison_results'), exist_ok=True)


def predict(model_name, model_pth, target_id, data_name):
    # root output dir
    root_pred_dir = rf'D:\home\school\ntut\dataset\chest-ct-segmentation\pred_{data_name}\{model_name}'

    # root src dir
    root_img_dir = fr'D:\home\school\ntut\dataset\chest-ct-segmentation\{data_name}\images'
    root_lab_dir = fr'D:\home\school\ntut\dataset\chest-ct-segmentation\{data_name}\masks'

    # image and label dir
    img_fd = target_id
    img_dir = os.path.join(root_img_dir, img_fd)
    lab_dir = os.path.join(root_lab_dir, img_fd)

    # make root output dir
    mk_predict_dir(root_pred_dir)

    # set output dir path
    inp_dir = os.path.join(root_pred_dir, 'images', img_fd)
    mask_dir = os.path.join(root_pred_dir, 'masks', img_fd)
    seg_dir = os.path.join(root_pred_dir, 'segs', img_fd)
    mask_result_dir = os.path.join(root_pred_dir, 'mask_results', img_fd)
    gt_dir = os.path.join(root_pred_dir, 'gts', img_fd)
    gt_result_dir = os.path.join(root_pred_dir, 'gt_results', img_fd)
    comparison_result_dir = os.path.join(root_pred_dir, 'comparison_results', img_fd)

    # make output dir
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(mask_result_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(gt_result_dir, exist_ok=True)
    os.makedirs(comparison_result_dir, exist_ok=True)

    # set data path
    x_fs = sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    y_fs = sorted(os.listdir(lab_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    x_pths = [os.path.join(img_dir, x_f) for x_f in x_fs]
    y_pths = [os.path.join(lab_dir, y_f) for y_f in y_fs]

    # load dataset
    ds = TestDataset(x_pths, y_pths)

    # init model
    net = Net(model_pth)

    # get item form dataset
    for i, (img, lab) in enumerate(tqdm(ds)):
        # predict
        pred = np.uint8(net.predict(img))

        # src image
        inp_img = Image.fromarray(rescale_img(img)).convert("L")
        lab_img = Image.fromarray(rescale_img(lab)).convert("L")

        # pred imgs
        mask_img = Image.fromarray(rescale_img(pred)).convert("L")
        seg_img = Image.fromarray(np.array(inp_img) * pred).convert("L")

        # save imgs
        inp_img.save(os.path.join(inp_dir, f'{i}.bmp'))
        mask_img.save(os.path.join(mask_dir, f'{i}.bmp'))
        seg_img.save(os.path.join(seg_dir, f'{i}.bmp'))
        lab_img.save(os.path.join(gt_dir, f'{i}.bmp'))
        write_pred_mask(inp_img, mask_img, os.path.join(mask_result_dir, f'{i}.bmp'))
        write_pred_mask(inp_img, lab_img, os.path.join(gt_result_dir, f'{i}.bmp'))
        write_pred_label_mask(inp_img, mask_img, lab_img, os.path.join(comparison_result_dir, f'{i}.bmp'))


def predict_all(model_name, model_pth, data_name):
    img_dir = fr'D:\home\school\ntut\dataset\chest-ct-segmentation\{data_name}\images'
    target_ids = os.listdir(img_dir)
    for i, target_id in enumerate(target_ids):
        print(f'[{i}]: {target_id}')
        print(target_id)
        predict(model_name, model_pth, target_id, data_name)


if __name__ == '__main__':
    # ID00423637202312137826377, ID00367637202296290303449, ID00426637202313170790466, ID00007637202177411956430
    # ID00032637202181710233084
    target_id = 'ID00423637202312137826377'
    model_pth = '../model/adj_contrast/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs4_224/model_bst_ep100.pth'
    model_name = 'adj_contrast_model_bst_ep100'
    data_name = 'adj_contrast_data'

    # predict single
    # print(target_id)
    # predict(model_name, model_pth, target_id, data_name)

    # predict all
    predict_all(model_name, model_pth, data_name)
