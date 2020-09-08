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


# Draw bounding boxes
def draw_boxes(image, bounds, color='green', width=2):

    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image
