import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from TransUNet.utils import DiceLoss
from torchvision import transforms


def test_step(dataloader, model, dice_loss, ce_loss, tt_iter_num, writer, iterator):
    model.eval()
    loss_ls = []
    for sampled_batch in dataloader:
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        iterator.set_postfix_str('val iteration %d : loss : %f, loss_ce: %f' % (tt_iter_num, loss.item(), loss_ce.item()))
        loss_ls.append(loss.item())
        writer.add_scalar('info/tt_total_loss', loss, tt_iter_num)
        writer.add_scalar('info/tt_loss_ce', loss_ce, tt_iter_num)

        tt_iter_num += 1

    avg_loss = np.array(loss_ls).mean()

    return avg_loss, tt_iter_num


def trainer_heart(args, model, snapshot_path):
    from utils.dataset import HeartDataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    db_train = HeartDataset(x_dir=args.x_dir, y_dir=args.y_dir,
                            data_csv=args.tr_csv,
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_test = HeartDataset(x_dir=args.x_dir, y_dir=args.y_dir,
                            data_csv=args.tt_csv, is_test=True)

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    tt_iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch))
    min_loss = 1000
    for epoch_num in iterator:
        model.train()
        loss_ls = []

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            iterator.set_postfix_str('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            loss_ls.append(loss.item())

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        avg_loss = np.array(loss_ls).mean()
        logging.info(f"avg loss: {avg_loss}")

        if avg_loss < min_loss:
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"[epoch_{epoch_num}] save model to {save_mode_path}")
            min_loss = avg_loss

        if epoch_num % 10 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"save model to {save_mode_path}")

        tt_avg_loss, tt_iter_num = test_step(testloader, model, dice_loss, ce_loss, tt_iter_num, writer, iterator)
        logging.info(f"tt avg loss: {tt_avg_loss}")

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'last_epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
