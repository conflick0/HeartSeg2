import numpy as np
from PIL import Image

class Dataset:
    def __init__(self, x_pths, y_pths):
        self.x_pths = x_pths
        self.y_pths = y_pths

    def __len__(self):
        return len(self.x_pths)

    def __getitem__(self, idx):
        x_pth = self.x_pths[idx]
        x = load_img(x_pth)

        y_pth = self.y_pths[idx]
        y = load_img(y_pth)[:, :, 1]
        y = remove_artifacts(y)

        return x, y


def load_img(path):
    return np.array(Image.open(path))


def remove_artifacts(mask):
    mask[mask < 240] = 0  # remove artifacts
    mask[mask >= 240] = 255
    return mask