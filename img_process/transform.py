import numpy as np


def rescale(img):
    '''
    Rescale image to [0, 255]
    '''
    img = np.array(img)
    if np.max(img) > np.min(img):
        arr = img.astype(np.float32)
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        arr = (arr * 255).astype(np.uint8)
        return arr
    return (img * 0).astype(np.uint8)


def rescale_to_float(img):
    '''
    Rescale image to [0, 1]
    '''
    return (np.array(img) / 255.).astype(np.float32)

