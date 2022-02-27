import os
from os import path, listdir

import pandas as pd


def get_file_names(p_dir):
    '''get file names by patient dir'''
    return sorted(listdir(p_dir), key=lambda file_name: int(file_name.split('_')[-1].split('.')[0]))


def get_file_paths(dir, pid):
    '''get file get_file_paths by img/lab dir and patient id'''
    p_dir = path.join(dir, pid)
    return map(lambda file_name: path.join(pid, file_name), get_file_names(p_dir))


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_df(root_img_dir, root_lab_dir, pids):
    '''get training/test dataset dataframe'''
    img_paths = flatten(map(lambda pid: get_file_paths(root_img_dir, pid), pids))
    lab_paths = flatten(map(lambda pid: get_file_paths(root_lab_dir, pid), pids))

    df = pd.DataFrame({
        'ImageId': img_paths,
        'MaskId': lab_paths
    })

    return df


if __name__ == '__main__':
    data_name = 'adj_contract'
    root_dir = fr'D:\home\school\ntut\dataset\chest-ct-segmentation\{data_name}_data'
    root_img_dir = path.join(root_dir, 'images')
    root_lab_dir = path.join(root_dir, 'masks')

    csv_dir = f'../data_csv/{data_name}'
    tr_csv_pth = path.join(csv_dir, 'tr.csv')
    tt_csv_pth = path.join(csv_dir, 'tt.csv')

    # get pids
    pids = listdir(root_img_dir)

    # split tr and tt by pid nums
    num_tr_pid = int(len(pids) * 0.9)
    tr_pids = pids[:num_tr_pid]
    tt_pids = pids[num_tr_pid:]

    # show num of pid
    print(f'total num of pids: {len(pids)}')
    print(f'tr num of pids: {len(tr_pids)}')
    print(f'tt num of pids: {len(tt_pids)}')

    # get dataset img paths df
    tr_df = get_df(root_img_dir, root_lab_dir, tr_pids)
    tt_df = get_df(root_img_dir, root_lab_dir, tt_pids)

    # save to csv
    os.makedirs(csv_dir, exist_ok=True)
    tr_df.to_csv(tr_csv_pth, index=False)
    tt_df.to_csv(tt_csv_pth, index=False)

    # show num of imgs
    print(f'tr num of imgs:{len(tr_df)}')
    print(f'tt num of imgs:{len(tt_df)}')
