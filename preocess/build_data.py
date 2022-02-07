import os
from PIL import Image
from tqdm import tqdm

from preocess.dataset import Dataset

root_src_img_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\test_data\images'
root_src_lab_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\test_data\masks'

root_dst_img_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\clear_data\images'
root_dst_lab_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\clear_data\masks'

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

    ds = Dataset(x_pths, y_pths)

    for i, (x, y) in enumerate(ds):
        x_im = Image.fromarray(x).convert("L")
        y_im = Image.fromarray(y).convert("L")

        x_im.save(os.path.join(dst_img_dir, f'{pid}_{i}.jpg'))
        y_im.save(os.path.join(dst_lab_dir, f'{pid}_mask_{i}.jpg'))


