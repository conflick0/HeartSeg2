import os
import numpy as np
import nrrd
from PIL import Image


def write_nrrd(file_name, imgs):
    imgs = (imgs.T / 255.).astype(np.float32)

    header = {
        'space directions': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'space origin': [0, 0, 0]
    }

    nrrd.write(file_name, imgs, header)


def write_pred_nrrd(img_dir, file_name, is_raw=False):
    img_fs = sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0]))

    if is_raw:
        imgs = np.array([np.array(Image.open(os.path.join(img_dir, img_f)).resize((512, 512))) for img_f in img_fs])
    else:
        imgs = np.array([np.array(Image.open(os.path.join(img_dir, img_f))) for img_f in img_fs])
        imgs = np.flip(imgs, 0)  # reverse

    write_nrrd(file_name, imgs)


def write_raw_data(target, pred_seg_dir, pred_mask_dir):
    write_pred_nrrd(pred_seg_dir, f'data/nrrd_data/seg_{target}.nrrd', is_raw=True)
    write_pred_nrrd(pred_mask_dir, f'data/nrrd_data/mask_{target}.nrrd', is_raw=True)


def write_test_data(target, pred_seg_dir, pred_mask_dir, pred_gt_mask_dir):
    write_pred_nrrd(pred_seg_dir, f'data/nrrd_data/seg_{target}.nrrd')
    write_pred_nrrd(pred_mask_dir, f'data/nrrd_data/mask_{target}.nrrd')
    write_pred_nrrd(pred_gt_mask_dir, f'data/nrrd_data/gt_mask_{target}.nrrd')


if __name__ == '__main__':
    # ID00423637202312137826377, ID00367637202296290303449, raw_dcm
    # target = 'ID00423637202312137826377'
    # pred_seg_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_data\segs\{target}'
    # pred_mask_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_data\masks\{target}'
    # pred_gt_mask_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_data\gt_masks\{target}'

    # write_test_data(target, pred_seg_dir, pred_mask_dir, pred_gt_mask_dir)

    # target = 'raw_dcm'
    # pred_seg_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_data\segs\{target}'
    # pred_mask_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_data\masks\{target}'
    # pred_gt_mask_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_data\gt_masks\{target}'

    target = 'raw_dcm'
    pred_seg_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_crop_data\segs\{target}'
    pred_mask_dir = rf'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_crop_data\masks\{target}'

    write_raw_data(target, pred_seg_dir, pred_mask_dir)
