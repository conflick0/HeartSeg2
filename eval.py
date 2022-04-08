import os
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


if __name__ == '__main__':

    for n in ['crop2']:
        data_name = n
        tt_csv_pth = f'data_csv/{data_name}/tt.csv'
        pids = get_pid_from_file(tt_csv_pth)
        data_dir = fr'D:\dataset\chest-ct-segmentation\{data_name}_data'
        model_pth = f'model/{data_name}/TU_pretrain_R50-ViT-B_16_skip3_epo100_bs4_224/model_bst_ep100.pth'
        out_dir = f'data_csv/eval/{data_name}'

        mc_df = eval_patients(pids, data_dir, model_pth)

    os.makedirs(out_dir, exist_ok=True)
    mc_df.to_csv(os.path.join(out_dir, 'tt.csv'), index=False)
