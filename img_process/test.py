from img_process import convert as cvt


def test_dcm2img():
    dcm_dir = r'D:\home\school\ntut\dataset\corcta\corcta_dcm'
    img_dir = r'C:\Users\jack\Downloads\dcm_imgs'
    cvt.dcm2img(dcm_dir, img_dir, 'bmp')


def test_nrrd2img():
    nrrd_path = r'C:\Users\jack\Downloads\Segmentation.nrrd'
    nrrd_path1 = r'C:\Users\jack\Downloads\Segmentation1.nrrd'
    img_dir = r'C:\Users\jack\Downloads\nrrd_imgs'
    img_dir1 = r'C:\Users\jack\Downloads\nrrd_imgs1'
    cvt.nrrd2img(nrrd_path, img_dir, 'bmp')
    cvt.img2nrrd(img_dir, nrrd_path1)
    cvt.nrrd2img(nrrd_path1, img_dir1, 'bmp')


def test_ptn2img():
    ptn_path = r'C:\Users\jack\Downloads\segment_512_512_267.ptn'
    ptn_path1 = r'C:\Users\jack\Downloads\segment1_512_512_267.ptn'
    img_dir = r'C:\Users\jack\Downloads\ptn_imgs'
    img_dir1 = r'C:\Users\jack\Downloads\ptn_imgs1'
    cvt.ptn2img(ptn_path, img_dir, (512, 512, 267), 'bmp')
    cvt.img2ptn(img_dir, ptn_path1)
    cvt.ptn2img(ptn_path1, img_dir1, (512, 512, 267), 'bmp')


if __name__ == '__main__':
    test_dcm2img()
    test_nrrd2img()
    test_ptn2img()
