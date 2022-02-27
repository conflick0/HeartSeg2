import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class HeartDataset(Dataset):
    def __init__(self, x_dir, y_dir, data_csv, is_test=False, transform=None):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.df = pd.read_csv(data_csv)
        self.is_test = is_test
        self.transform = transform  # using transform in torch!

    def __len__(self):
        return len(self.df['ImageId'])

    def __getitem__(self, idx):
        x_pth = os.path.join(self.x_dir, self.df.iloc[idx, 0])
        y_pth = os.path.join(self.y_dir, self.df.iloc[idx, 1])

        image = normalise_zero_one(np.array(Image.open(x_pth)))

        label = np.array(Image.open(y_pth))

        if len(label.shape) == 3:
            label = label[:, :, 1]

        label = normalise_zero_one(
            remove_artifacts(label)
        )

        im_x, im_y = image.shape
        lb_x, lb_y = label.shape

        image = zoom(image, (224 / im_x, 224 / im_y), order=3)
        label = zoom(label, (224 / lb_x, 224 / lb_y), order=0)

        if self.is_test:
            image = np.stack((image, image, image), 0)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = x_pth.split('\\')[-1].split('_')[0]

        return sample


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def remove_artifacts(mask):
    mask[mask < 240] = 0  # remove artifacts
    mask[mask >= 240] = 255
    return mask


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
