import numpy as np
from PIL import Image
import os


def remove_artifacts(mask):
    mask[mask < 240] = 0  # remove artifacts
    mask[mask >= 240] = 255
    return mask


def read_ptn(fn, idx=0):
    bytes = np.fromfile(fn, dtype=np.uint8)
    bits = np.unpackbits(bytes, bitorder='little')
    a = (bits * 255).astype(np.uint8).reshape((512, 512, 267), order='F')
    # print(arr.shape)
    img = Image.fromarray(a[:, :, idx].T, mode='L')
    img.show()


def write_ptn(inp_dir, out_fns):
    fns = sorted(os.listdir(inp_dir), key=lambda fn: int(fn.split('.')[0]))
    arrs = []
    for fn in fns:
        img = Image.open(os.path.join(inp_dir, fn)).convert('L')
        img = Image.fromarray(remove_artifacts(np.array(img)))
        arr = np.array(img).T
        arrs.append(arr)
    arrs = np.moveaxis(np.array(arrs), 0, -1).flatten(order='F')
    np.packbits(arrs, bitorder='little').astype(np.uint8).tofile(out_fns)
    # np.save(out_fns, bytes)


if __name__ == '__main__':
    # input_file = r'C:\Users\cg\Desktop\cardiac_models\model_4\segment_512_512_267.ptn'
    # read_ptn(input_file)
    input_dir = r'D:\project\3DSlicerProject\corcta_project\v4\mask\jpg'
    out_file = r'D:\project\3DSlicerProject\corcta_project\v4\mask\segment_512_512_267.ptn'
    write_ptn(input_dir, out_file)
    # read_ptn(out_file, idx=200)
