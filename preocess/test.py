import os


root_dst_img_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\clear_data\images'
root_dst_lab_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\clear_data\masks'

pids = sorted(os.listdir(root_dst_img_dir))


tr_pids = pids[int(len(pids)*0.8):]
all_x_pths = []
for tr_pid in tr_pids:
    fd = os.path.join(root_dst_img_dir, tr_pid)
    x_pths = [os.path.join(fd, f) for f in sorted(os.listdir(fd), key=lambda x: int(x.split('_')[-1].split('.')[0]))]
    all_x_pths.extend(x_pths)

print(len(all_x_pths))
print(all_x_pths[-1])
print(len(tr_pids))

