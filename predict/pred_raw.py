import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from predict.datasets.pad_raw_dataset import PadRawDataset
from predict.nets.net import Net
from predict.utils.img_loader import rescale_img, threshold_img, show_pred_mask, write_pred_mask


def mk_predict_dir(dir):
    os.makedirs(os.path.join(dir), exist_ok=True)
    os.makedirs(os.path.join(dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'segs'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'mask_results'), exist_ok=True)


def predict(target_id, threshold_ratio=0.6):
    # root output dir
    root_pred_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\pred_data'

    # root src dir
    img_dir = r'D:\home\school\ntut\lab\dataset\corcta_dcm'

    # image and label dir
    img_fd = target_id

    # make root output dir
    mk_predict_dir(root_pred_dir)

    # set output dir path
    inp_dir = os.path.join(root_pred_dir, 'images', img_fd)
    mask_dir = os.path.join(root_pred_dir, 'masks', img_fd)
    seg_dir = os.path.join(root_pred_dir, 'segs', img_fd)
    mask_result_dir = os.path.join(root_pred_dir, 'mask_results', img_fd)

    # make output dir
    os.makedirs(inp_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(mask_result_dir, exist_ok=True)

    # set data path
    x_fs = sorted(os.listdir(img_dir))
    x_pths = [os.path.join(img_dir, x_f) for x_f in x_fs]

    # load dataset
    ds = PadRawDataset(x_pths)

    # init model
    net = Net(model_pth)

    # get item form dataset
    for i, img in enumerate(tqdm(ds)):
        # predict
        pred = net.predict(img)

        # threshold
        th_pred = threshold_img(pred, threshold_ratio)

        # src image
        inp_img = Image.fromarray(rescale_img(img)).convert("L")

        # pred imgs
        mask_img = Image.fromarray(rescale_img(th_pred)).convert("L")
        seg_img = Image.fromarray(np.array(inp_img) * th_pred).convert("L")

        # save imgs
        inp_img.save(os.path.join(inp_dir, f'{i}.jpg'))
        mask_img.save(os.path.join(mask_dir, f'{i}.jpg'))
        seg_img.save(os.path.join(seg_dir, f'{i}.jpg'))
        write_pred_mask(inp_img, mask_img, os.path.join(mask_result_dir, f'{i}.jpg'))


if __name__ == '__main__':
    model_pth = '../model/model.pth'
    target_id = 'raw_dcm'
    print(target_id)
    predict(target_id, threshold_ratio=0.75)
