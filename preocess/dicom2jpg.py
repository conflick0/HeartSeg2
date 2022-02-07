import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from predict.utils.img_loader import load_dcm

dcm_dir = r'D:\home\school\ntut\lab\dataset\corcta_dcm'
jpg_dir = r'D:\home\school\ntut\lab\dataset\corcta_dcm_jpg'

os.makedirs(jpg_dir, exist_ok=True)

fns = os.listdir(dcm_dir)

for fn in tqdm(fns):
    img_pth = os.path.join(dcm_dir, fn)
    img = load_dcm(img_pth)
    img = Image.fromarray(img).convert("L")

    out_pth = os.path.join(jpg_dir, f"{fn.split('.')[0]}.jpg")
    img.save(out_pth)
