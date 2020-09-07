import cv2
import fire
import numpy as np


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img = cv2.filter2D(img, -1, kernel=kernel)
    return img


def get_not_poly_idx(is_polys):
    return [i for i, x in enumerate(is_polys) if x == False]


def filter_not_poly_img(cropped_imgs, not_poly_idxs):
    return [i for j, i in enumerate(cropped_imgs) if j not in not_poly_idxs]


def text_dectection_preprocess(imgs, size=720, do_sharpen=True):
    process_imgs = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if do_sharpen:
            img = sharpen(img)
        process_imgs.append(img)

    return process_imgs


def main(yaml='args/detection_args.yaml',
         imgs=['image/TA01_24.tif', ],
         ):

    from imgcat import imgcat
    ori_imgs = []
    for img in imgs:
        ori_imgs.append(cv2.imread(img))
    process_imgs = text_dectection_preprocess(ori_imgs)
    imgcat(process_imgs[0])

    # breakpoint()
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    fire.Fire(main)
