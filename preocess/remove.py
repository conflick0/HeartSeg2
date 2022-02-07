import csv
import os.path
import shutil

root_dst_img_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\clear_data\images'
root_dst_lab_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\clear_data\masks'

with open('../data_csv/rm_pid.csv', newline='', encoding='utf-8-sig') as f:
    rows = csv.reader(f)
    for row in rows:
        print(row)
        shutil.rmtree(os.path.join(root_dst_img_dir, row[0]))
        shutil.rmtree(os.path.join(root_dst_lab_dir, row[0]))