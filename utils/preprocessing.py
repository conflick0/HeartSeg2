from config import config
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

    def _get_pid_df(self):
        pids = self.df['ImageId'].apply(lambda x: x.split("_")[0]).reset_index(name='PatientId')
        return pids.groupby('PatientId').size().reset_index(name='ImageNum')

    def split_data(self):
        tr_num = int(len(self.df) * 0.2)

        name = 'A'
        tt_df = self.df.iloc[0:tr_num, :]
        tr_df = self.df.iloc[tr_num:, :]
        tr_df.to_csv(f"../data_csv/{name}/tr_{name}.csv", index=False)
        tt_df.to_csv(f"../data_csv/{name}/tt_{name}.csv", index=False)

        for i, name in enumerate(['B', 'C', 'D'], 1):
            print(i)
            tr_df1 = self.df.iloc[0:tr_num*i, :]
            tt_df = self.df.iloc[tr_num*i:tr_num*(i+1), :]
            tr_df2 = self.df.iloc[tr_num*(i+1):, :]
            tr_df = pd.concat([tr_df1, tr_df2], axis=0)

            tr_df.to_csv(f"../data_csv/{name}/tr_{name}.csv", index=False)
            tt_df.to_csv(f"../data_csv/{name}/tt_{name}.csv", index=False)

        name = 'E'
        tr_df = self.df.iloc[0:tr_num*4, :]
        tt_df = self.df.iloc[tr_num*4:, :]
        tr_df.to_csv(f"../data_csv/{name}/tr_{name}.csv", index=False)
        tt_df.to_csv(f"../data_csv/{name}/tt_{name}.csv", index=False)


if __name__ == '__main__':
    dp = DataPath(config.raw_data_csv)


