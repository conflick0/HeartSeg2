import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class config:
    inp_dir = r'D:/home/school/ntut/dataset/chest-ct-segmentation/crop2_data/images'
    lab_dir = r'D:/home/school/ntut/dataset/chest-ct-segmentation/crop2_data/masks'
    raw_data_csv = r'D:/home/school/ntut/dataset/chest-ct-segmentation/raw_data/train.csv'
    tr_csv = r'D:\home\school\ntut\project\HeartSeg\data_csv\crop2\tr.csv'
    tt_csv = r'D:\home\school\ntut\project\HeartSeg\data_csv\crop2\tt.csv'
    model_pth = os.path.join(ROOT_DIR, 'model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz')
    model_weight_pth = os.path.join(ROOT_DIR, 'model/eval_model/crop/epoch_9.pth')
    max_epochs = 10
    batch_size = 4
