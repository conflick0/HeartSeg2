import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from predict.datasets.raw_dataset import RawDataset
from predict.nets.net import Net
from predict.utils.img_loader import rescale_img, threshold_img, show_pred_mask, write_pred_mask


def mk_predict_dir(dir):
    os.makedirs(os.path.join(dir), exist_ok=True)
    os.makedirs(os.path.join(dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'segs'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'mask_results'), exist_ok=True)


def predict(target_id, model_name):
    # root output dir
    root_pred_dir = fr'D:\home\school\ntut\dataset\chest-ct-segmentation\pred_crop_data\{model_name}'

    # root src dir
    img_dir = r'D:\home\school\ntut\dataset\corcta\corcta_dcm'

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
    ds = RawDataset(x_pths)

    # init model
    net = Net(model_pth)

    # get item form dataset
    for i, img in enumerate(tqdm(ds)):
        # predict
        pred = np.uint8(net.predict(img))

        # src image
        inp_img = Image.fromarray(rescale_img(img)).convert("L")

        # pred imgs
        mask_img = Image.fromarray(rescale_img(pred)).convert("L")
        seg_img = Image.fromarray(np.array(inp_img) * pred).convert("L")

        # save imgs
        inp_img.save(os.path.join(inp_dir, f'{i}.bmp'))
        mask_img.save(os.path.join(mask_dir, f'{i}.bmp'))
        seg_img.save(os.path.join(seg_dir, f'{i}.bmp'))
        write_pred_mask(inp_img, mask_img, os.path.join(mask_result_dir, f'{i}.bmp'))


if __name__ == '__main__':
    model_pth = '../model/crop2/model_99.pth'
    model_name = 'crop2_model_99'
    target_id = 'raw_dcm'
    print(target_id)
    predict(target_id, model_name)
