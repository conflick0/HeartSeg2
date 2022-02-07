import os
import shutil
import pandas as pd


def get_df_by_pid(df, pid):
    idxs = df['ImageId'].apply(lambda x: x.split("_")[0]).loc[lambda x: x == pid].index
    return df.loc[idxs, :]


class DataPath:
    def __init__(self, data_csv):
        self.df = pd.read_csv(data_csv)
        self.img_ids = self.df['ImageId']
        self.lab_ids = self.df['MaskId']
        self.pid_df = self._get_pid_df()

    def show_patient_info(self, idx):
        print(self.pid_df.iloc[idx, :])

    def read_by_pid(self, pid):
        is_target = self.df['PatientId'] == pid
        return self.df[is_target]

    def _get_pid_df(self):
        self.df['PatientId'] = self.df['ImageId'].apply(lambda x: x.split("_")[0])
        return pd.DataFrame(self.df.groupby('PatientId').size(), columns=['Size']).reset_index()

    def split_data(self):
        tr_num = int(len(self.df) * 0.2)

        name = 'A'
        tt_df = self.df.iloc[0:tr_num, :]
        tr_df = self.df.iloc[tr_num:, :]
        tr_df.to_csv(f"../data_csv/{name}/tr_{name}.csv", index=False)
        tt_df.to_csv(f"../data_csv/{name}/tt_{name}.csv", index=False)

        for i, name in enumerate(['B', 'C', 'D'], 1):
            print(i)
            tr_df1 = self.df.iloc[0:tr_num * i, :]
            tt_df = self.df.iloc[tr_num * i:tr_num * (i + 1), :]
            tr_df2 = self.df.iloc[tr_num * (i + 1):, :]
            tr_df = pd.concat([tr_df1, tr_df2], axis=0)

            tr_df.to_csv(f"../data_csv/{name}/tr_{name}.csv", index=False)
            tt_df.to_csv(f"../data_csv/{name}/tt_{name}.csv", index=False)

        name = 'E'
        tr_df = self.df.iloc[0:tr_num * 4, :]
        tt_df = self.df.iloc[tr_num * 4:, :]
        tr_df.to_csv(f"../data_csv/{name}/tr_{name}.csv", index=False)
        tt_df.to_csv(f"../data_csv/{name}/tt_{name}.csv", index=False)


if __name__ == '__main__':
    inp_dir = r'D:/home/school/ntut/lab/dataset/chest-ct-segmentation/raw_data/images'
    lab_dir = r'D:/home/school/ntut/lab/dataset/chest-ct-segmentation/raw_data/masks'
    raw_data_csv = r'D:/home/school/ntut/lab/dataset/chest-ct-segmentation/raw_data/train.csv'
    root_dir = r'D:\home\school\ntut\lab\dataset\chest-ct-segmentation\test_data'

    dst_inp_dir = os.path.join(root_dir, 'images')
    dst_lab_dir = os.path.join(root_dir, 'masks')
    os.mkdir(dst_inp_dir)
    os.mkdir(dst_lab_dir)

    dp = DataPath(raw_data_csv)
    pids = dp.pid_df['PatientId']

    for pid in pids:
        print(pid)
        df = dp.read_by_pid(pid)

        pid_inp_dir = os.path.join(dst_inp_dir, pid)
        pid_lab_dir = os.path.join(dst_lab_dir, pid)
        os.mkdir(pid_inp_dir)
        os.mkdir(pid_lab_dir)

        for x, y in zip(df['ImageId'], df['MaskId']):
            x_pth = os.path.join(inp_dir, x)
            y_pth = os.path.join(lab_dir, y)

            shutil.copy2(x_pth, pid_inp_dir)
            shutil.copy2(y_pth, pid_lab_dir)