import pydicom
from PIL import Image
import numpy as np
import nrrd
import os
from tqdm import tqdm
from img_process.transform import rescale, rescale_to_float


def sorted_func(file_name):
    """
    Returns a sorted function.
    """
    basename = file_name.split('.')[0]
    if basename.find('_') != -1:
        return int(basename.split('_')[-1])
    elif basename.find('-') != -1:
        return basename
    else:
        return int(basename)


def list_sorted_dir(dir):
    """
    Returns a list of sorted files by file name.
    """
    return sorted(os.listdir(dir), key=lambda fn: sorted_func(fn))


def write_imgs(imgs, img_dir, save_img_type='jpg'):
    """
    Save imgs to img_dir.
    param: imgs: imgs
    param: img_dir: img dir
    param: save_img_type: save img type,
        {'jpg', 'bmp'}, default is 'jpg'
    """
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(tqdm(imgs)):
        img_pth = os.path.join(img_dir, f"{i}.{save_img_type}")
        img.save(img_pth)


def read_imgs(img_dir):
    """
    Read imgs from dir.
    param: img_dir: img dir
    return: imgs
    """
    img_fs = list_sorted_dir(img_dir)
    imgs = []
    for img_f in tqdm(img_fs):
        img_pth = os.path.join(img_dir, img_f)
        img = Image.open(img_pth)
        imgs.append(img)
    return imgs


def read_dcm(path):
    """
    Reads a dicom file and returns the image.
    param: path: path to the dicom file
    """
    dcm = pydicom.dcmread(path)
    arr = rescale(dcm.pixel_array)
    return Image.fromarray(arr)


def read_dcms(dir):
    """
    Reads a dicom files and returns the list of images.
    param: path: path to the dicom file
    """
    files = os.listdir(dir)
    imgs = []
    for i, file in enumerate(files):
        dcm_path = os.path.join(dir, file)
        img = read_dcm(dcm_path)
        imgs.append(img)
    return imgs


def read_nrrd(path):
    """
    Reads a nrrd file and returns the list of images.
    param: path: path to the nrrd file
    """
    arr, _ = nrrd.read(path)
    arr = np.flip(arr.T, axis=0)
    imgs = list(map(lambda img: Image.fromarray(rescale(img)), arr))
    return imgs


def write_nrrd(images, path):
    """
    Writes a nrrd file from a list of images.
    param: images: list of images
    param: path: path to the nrrd file
    """
    header = {
        'space directions': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        'space origin': [0, 0, 0]
    }
    arr = np.array(list(map(lambda img: np.array(rescale_to_float(img)), images)))
    arr = np.flip(arr.T, axis=-1)
    nrrd.write(path, arr, header)


def read_ptn(path, shape):
    """
    Reads a ptn file and returns the list of images.
    param: path: path to the ptn file
    param: shape: shape of the ptn file (3d array)
    """
    bytes = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(bytes, bitorder='little')
    arr = (bits * 255).astype(np.uint8).reshape(shape, order='F')
    imgs = list(map(Image.fromarray, arr.T))
    return imgs


def write_ptn(images, path):
    """
    Writes a ptn file from a list of images.
    param: images: list of images
    param: path: path to the ptn file
    """
    arr = np.array(list(map(lambda img: np.array(img).T, images))).astype(np.uint8)
    flatten_arr = np.moveaxis(arr, 0, -1).flatten(order='F')
    bits = np.packbits(flatten_arr, bitorder='little')
    bits.tofile(path)

