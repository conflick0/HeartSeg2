import torch
import numpy as np
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os

ROOT_DIR = r'D:\home\school\ntut\project\HeartSeg2'

model_pth = os.path.join(ROOT_DIR, 'model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz')
model_weight_pth = os.path.join(ROOT_DIR, 'model/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs4_224/bst_ep100.pth')
out_model_pth = '../model/crop2/model_bst_ep100.pth'
vit_name = 'R50-ViT-B_16'
num_classes = 2
n_skip = 3

config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = num_classes
config_vit.n_skip = n_skip
img_size = 224
vit_patches_size = 16

if vit_name.find('R50') != -1:
    config_vit.patches.grid = (
        int(img_size / vit_patches_size), int(img_size / vit_patches_size))
net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
config_vit.pretrained_path = model_pth
net.load_from(weights=np.load(config_vit.pretrained_path))

net.load_state_dict(torch.load(model_weight_pth))
torch.save(net, out_model_pth)
