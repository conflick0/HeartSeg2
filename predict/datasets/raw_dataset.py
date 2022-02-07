import os
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
from predict.utils.img_loader import load_dcm


class RawDataset(Dataset):
    def __init__(self, x_pths):
        self.x_pths = x_pths

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.x_pths)

    def __getitem__(self, idx):
        x_pth = self.x_pths[idx]
        x = Image.fromarray(np.uint8(load_dcm(x_pth)))
        x = self.transform(x)
        x = x.squeeze(0).numpy()

        return x
