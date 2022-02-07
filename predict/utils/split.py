import os
from img_loader import list_sorted_test_dir
import pandas as pd

def build_csv(pids, file_name):
    image_ids = []
    mask_ids = []
    for pid in pids:
        x_dir = os.path.join(dst_inp_dir, pid)
        y_dir = os.path.join(dst_lab_dir, pid)

        x_fs = list_sorted_test_dir(x_dir)
        y_fs = list_sorted_test_dir(y_dir)

        x_fs = [os.path.join(pid, x_f) for x_f in x_fs]
        y_fs = [os.path.join(pid, y_f) for y_f in y_fs]

        image_ids.extend(x_fs)
        mask_ids.extend(y_fs)

    df = pd.DataFrame({'ImageId': image_ids, 'MaskId': mask_ids})
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    dst_inp_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\crop_data\images'
    dst_lab_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\crop_data\masks'

    data_crop_csv_dir = r'D:\home\school\ntut\lab\project\HeartSeg\data_csv\crop'

    tr_csv = os.path.join(data_crop_csv_dir, 'tr.csv')
    tt_csv = os.path.join(data_crop_csv_dir, 'tt.csv')

    # split data to tr and tt
    pids = sorted(os.listdir(dst_inp_dir))
    tr_num = int(len(pids) * 0.8)
    tr_pids = pids[:tr_num]
    tt_pids = pids[tr_num:]

    # build img file name csv
    build_csv(tr_pids, tr_csv)
    build_csv(tt_pids, tt_csv)