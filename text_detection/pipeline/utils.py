import io
import json
from collections import OrderedDict
from pathlib import Path
from time import strftime

import cv2
import numpy as np
import yaml
from easydict import EasyDict
from PIL import Image, ImageDraw


def load_yaml(config_path, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):

    config_path = str(config_path)

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    with open(config_path, 'r') as f:
        output = yaml.load(f, OrderedLoader)
    return output


def load_ocr_map(yaml_path):
    args_map = load_yaml(yaml_path)
    args_map = json.loads(json.dumps(args_map))
    args_map = {
        key: {
            key2: EasyDict(value2)
            for key2, value2 in value.items()
        }
        for key, value in args_map.items()
    }

    return args_map


def is_bytes(x):
    return isinstance(x, (bytes, bytearray))


def is_file(x):
    return Path(x).is_file()


def byte2img(imbyte):
    ensure_condition(
        condition=is_bytes(imbyte),
        info=f'The given imbyte {type(imbyte)} is not a byte.'
    )
    return Image.open(io.BytesIO(imbyte))


def check_polygon(polygon):
    is_poly = True
    polygon = np.array(polygon)
    s = polygon.sum(axis=1)
    if np.argmin(s) != 0:
        is_poly = False
    if np.argmax(s) != 2:
        is_poly = False

    diff = np.diff(polygon, axis=1)
    if np.argmin(diff) != 1:
        is_poly = False
    if np.argmax(diff) != 3:
        is_poly = False
    return is_poly


def warp_image(img, polygon):
    h, w = img.shape[:2]
    id_w = w
    id_h = int(w * 4.8 / 8)

    polygon = np.float32(polygon)

    dest = np.float32([
        (0,    0),
        (id_w, 0),
        (id_w, id_h),
        (0, id_h),
    ])

    m = cv2.getPerspectiveTransform(polygon, dest)
    perspective_image = cv2.warpPerspective(
        img, m, (id_w, id_h), flags=cv2.INTER_CUBIC)
    return perspective_image


def get_max_area(contours):
    areas = []
    for i in range(len(contours)):
        if len(contours[i]) < 3:
            continue
        area = cv2.contourArea(contours[i])
        areas.append(area)
    return max(areas) if areas else 0


def imdraw_seg(img, seg_points, color=[255, 255, 255]):
    ensure_condition(is_numpy_img(img), f'The type of img is not numpy')
    ensure_condition(is_tuple(seg_points) or is_list(seg_points))
    # process
    seg_points = np.array(seg_points).flatten().reshape(-1, 2).astype(int)
    img = cv2.fillConvexPoly(img.copy(), seg_points, color)
    return img


def save_ori_img(image):
    out_path = '../images/realworld_images'
    Path(out_path).mkdir(parents=True, exist_ok=True)
    imgname = 'ori_'+strftime("%Y_%m_%d_%H_%M_%S")+'.jpg'
    imgname = str(Path(out_path)/imgname)
    cv2.imwrite(imgname, image)


def save_box_img(image):
    out_path = '../images/box_images'
    Path(out_path).mkdir(parents=True, exist_ok=True)
    imgname = 'box_'+strftime("%Y_%m_%d_%H_%M_%S")+'.jpg'
    imgname = str(Path(out_path)/imgname)
    cv2.imwrite(imgname, image)


# Draw bounding boxes
def draw_boxes(image, bounds, color='green', width=2):

    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


def draw_boxes_char_level(image, bounds, color='green', width=2):

    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p2 = bound
        p1, p3 = (p2[0], p0[1]), (p0[0], p2[1])
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


def cut_images(image, bounds):
    cut_imgs = []
    for bound in bounds:
        bbox = (bound[0][0], bound[0][2])
        cut_imgs.append(cut(image, bbox))
    return cut_imgs


def cut(image, box):
    (xmin, ymin), (xmax, ymax) = box
    return image[ymin:ymax, xmin:xmax]


def is_list(x):
    return isinstance(x, list)


def is_numpy(x):
    return isinstance(x, np.ndarray)


def is_numpy_img(x):
    return is_numpy(x) and x.ndim == 2 or (x.ndim == 3 and x.shape[-1] in [1, 3])


def is_tuple(x):
    return isinstance(x, tuple)


def ensure_condition(condition, info):
    assert condition, info
