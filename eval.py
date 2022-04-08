import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataset import HeartDataset
from TransUNet.utils import calculate_metric_percase
from predict.datasets.test_dataset import TestDataset
from predict.nets.net import Net


def get_pid_from_file(file_path):
    """
    Get PID from file name.

    :param file_path: File path.
    :return: PID.
    """
    # read csv file using pandas
    df = pd.read_csv(file_path)

    # df extact the column 'IamgeId', and apply the function 'split' to it
    # the result is a list of strings
    # and unique() to remove the duplicated elements
    pid_list = df['ImageId'].apply(lambda x: x.split('\\')[0]).unique()
    return pid_list


def load_data_paths(img_dir, lab_dir):
    """
    Load data paths.

    :param img_dir: Iamge directory.
    :param lab_dir: Label directory.
    :return: Data paths.
    """
    # get all file names
    x_fs = sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    y_fs = sorted(os.listdir(lab_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    x_pths = [os.path.join(img_dir, x) for x in x_fs]
    y_pths = [os.path.join(lab_dir, y_f) for y_f in y_fs]

    return x_pths, y_pths


def eval_patient(img_dir, lab_dir, model_pth):
    # load data path
    x_pths, y_pths = load_data_paths(img_dir, lab_dir)

    # load dataset
    ds = TestDataset(x_pths, y_pths)

    # init model
    net = Net(model_pth)

    # get item form dataset
    dcs = []
    hd95s = []
    for i, (img, lab) in enumerate(tqdm(ds)):
        # predict
        pred = np.uint8(net.predict(img))
        dc, hd95 = calculate_metric_percase(pred, lab)
        dcs.append(dc)
        hd95s.append(hd95)

    return np.mean(dcs), np.mean(hd95s)


def eval_patients(pids, data_dir, model_pth):
    mc_df = pd.DataFrame(columns=['pid', 'dc', 'hd95'])
    for pid in pids:
        img_dir = os.path.join(data_dir, 'images', pid)
        lab_dir = os.path.join(data_dir, 'masks', pid)
        dc, hd95 = eval_patient(img_dir, lab_dir, model_pth)
        mc_df.loc[len(mc_df)] = [pid, dc, hd95]

    return mc_df


def eval_dataset(data_name, csv_pth, out_dir):
    pids = get_pid_from_file(csv_pth)
    data_dir = fr'D:\dataset\chest-ct-segmentation\{data_name}_data'
    model_pth = f'model/{data_name}/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs4_224/model_bst_ep100.pth'

    mc_df = eval_patients(pids, data_dir, model_pth)
    mc_df.to_csv(os.path.join(out_dir, Path(csv_pth).parts[-1]), index=False)


def eval_corcta_dataset(model_name, out_dir):
    model_pth = f'model/{model_name}/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs4_224/model_bst_ep100.pth'

    data_dir = 'D:\dataset\corcta\corcta_data'
    img_dir_names = ['corcta', 'corcta_adj_contract']
    lab_dir_name = 'mask'

    for img_dir_name in img_dir_names:
        img_dir = os.path.join(data_dir, img_dir_name)
        lab_dir = os.path.join(data_dir, lab_dir_name)
        dc, hd95 = eval_patient(img_dir, lab_dir, model_pth)

        mc_df = pd.DataFrame(columns=['pid', 'dc', 'hd95'])
        mc_df.loc[len(mc_df)] = [img_dir_name, dc, hd95]
        mc_df.to_csv(os.path.join(out_dir, f'{img_dir_name}.csv'), index=False)


if __name__ == '__main__':
    data_names = ['crop2', 'adj_contrast', 'adj_contrast_corcta']
    for data_name in data_names:
        print(f'{data_name} evaluation start.')
        base_csv_pth = f'data_csv/{data_name}'

        # make output directory
        out_dir = f'data_csv/eval/{data_name}'
        os.makedirs(out_dir, exist_ok=True)

        # eval train dataset
        print(f'{data_name} train evaluation start.')
        eval_dataset(data_name, os.path.join(base_csv_pth, 'tr.csv'), out_dir)

        # eval test dataset
        print(f'{data_name} test evaluation start.')
        eval_dataset(data_name, os.path.join(base_csv_pth, 'tt.csv'), out_dir)

        # eval corcta dataset
        print(f'{data_name} corcta evaluation start.')
        eval_corcta_dataset(data_name, out_dir)
