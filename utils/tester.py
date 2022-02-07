from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from TransUNet.utils import DiceLoss
import numpy as np


def tester_heart(x_dir, y_dir, data_csv, model, batch_size):
    from utils.dataset import HeartDataset
    num_classes = 2

    db_train = HeartDataset(x_dir, y_dir, data_csv, is_test=True)

    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    loss_ls = []
    loss_ce_ls = []

    iterator = tqdm(trainloader, ncols=70)

    for sampled_batch in iterator:
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        iterator.set_postfix_str(f'l:{loss.item():.2f},l_ce:{loss_ce.item():.2f}')

        loss_ls.append(loss.item())
        loss_ce_ls.append(loss_ce.item())

    iterator.close()

    return loss_ls, loss_ce_ls
