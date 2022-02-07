import torch
import numpy as np

from config import config
from utils.tester import tester_heart
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def load_model(model_weight_pth):
    vit_name = 'R50-ViT-B_16'
    img_size = 224
    vit_patches_size = 16
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = 2
    config_vit.n_skip = 3

    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    net = ViT_seg(config_vit, img_size, num_classes=config_vit.n_classes)
    config_vit.pretrained_path = config.model_pth
    net.load_from(weights=np.load(config_vit.pretrained_path))

    net.load_state_dict(torch.load(model_weight_pth))

    return net


def test(tt_csv, inp_dir, lab_dir, model_weight_pth):
    model = load_model(model_weight_pth).to('cuda')
    loss_ls, loss_ce_ls = tester_heart(inp_dir, lab_dir,
                                       tt_csv,
                                       model, batch_size=2)
    avg_loss = np.array(loss_ls).mean()
    avg_loss_ce = np.array(loss_ce_ls).mean()
    print(f'avg_loss: {avg_loss}, avg_loss_ce: {avg_loss_ce}')


if __name__ == '__main__':
    inp_dir = r'D:/home/school/ntut/lab/dataset/chest-ct-segmentation/crop_data/images'
    lab_dir = r'D:/home/school/ntut/lab/dataset/chest-ct-segmentation/crop_data/masks'
    tt_csv = r'D:\home\school\ntut\lab\project\HeartSeg\data_csv\crop\tt.csv'
    model_weight_pth = 'model/eval_model/crop/epoch_5.pth'
    test(tt_csv, inp_dir, lab_dir, model_weight_pth)