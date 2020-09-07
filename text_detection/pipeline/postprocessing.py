import math
import os

import cv2
import fire
import numpy as np
import torch
from PIL import Image

from text_detection.pipeline.detection import (do_detection,
                                               load_detection_model,
                                               load_detection_refined_model)
from text_detection.pipeline.preprocessing import text_dectection_preprocess
from text_detection.pipeline.utils import (draw_boxes, draw_boxes_char_level,
                                           load_ocr_map)


def easyocr_process(polys):
    result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return result


def polys_2_boxes(polys):
    boxes = []
    for poly in polys:
        xs, ys = poly.exterior.coords.xy
        xmin, xmax, ymin, ymax = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        # expand box
        dy = int((ymax - ymin) * 0.02)
        dx = int((xmax - xmin) * 0.02)
        xmin, xmax, ymin, ymax = xmin-dx, xmax+dx, ymin-dy, ymax+dy
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin

        boxes.append(
            [
                [xmin, ymin],
                [xmin, ymax],
                [xmax, ymax],
                [xmax, ymin],
            ]
        )

    return np.array(boxes).astype(int)


def extend_rect(image, box):
    h, w = image.shape[:2]
    (xmin, ymin), (xmax, ymax) = box
    ydelta = int((ymax - ymin)*0.2)
    xdelta = int((ymax - ymin)*0.2)
    ymin = ymin-ydelta if ymin-ydelta > 0 else 0
    ymax = ymax+ydelta if ymax+ydelta < h else h
    xmin = xmin-xdelta if xmin-xdelta > 0 else 0
    xmax = xmax+xdelta if xmax+xdelta < w else w

    return (xmin, ymin), (xmax, ymax)


def group_text_box(polys, args):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up = (poly[3]-poly[1])/np.maximum(10, (poly[2]-poly[0]))
        slope_down = (poly[5]-poly[7])/np.maximum(10, (poly[4]-poly[6]))
        if max(abs(slope_up), abs(slope_down)) < args.slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append(
                [x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min])
        else:
            height = np.linalg.norm([poly[6]-poly[0], poly[7]-poly[1]])
            margin = int(1.44*args.add_margin*height)

            theta13 = abs(
                np.arctan((poly[1]-poly[5])/np.maximum(10, (poly[0]-poly[4]))))
            theta24 = abs(
                np.arctan((poly[3]-poly[7])/np.maximum(10, (poly[2]-poly[6]))))
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13)*margin
            y1 = poly[1] - np.sin(theta13)*margin
            x2 = poly[2] + np.cos(theta24)*margin
            y2 = poly[3] - np.sin(theta24)*margin
            x3 = poly[4] + np.cos(theta13)*margin
            y3 = poly[5] + np.sin(theta13)*margin
            x4 = poly[6] - np.cos(theta24)*margin
            y4 = poly[7] + np.sin(theta24)*margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:

        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if (abs(np.mean(b_height) - poly[5]) < args.height_ths*np.mean(b_height)) and (abs(np.mean(b_ycenter) - poly[4]) < args.ycenter_ths*np.mean(b_height)):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(args.add_margin*box[5])
            merged_list.append(
                [box[0]-margin, box[1]+margin, box[2]-margin, box[3]+margin])
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if abs(box[0]-x_max) < args.width_ths * (box[3]-box[2]):  # merge boxes
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    margin = int(args.add_margin*(y_max - y_min))

                    merged_list.append(
                        [x_min-margin, x_max+margin, y_min-margin, y_max+margin])
                else:  # non adjacent box in same line
                    box = mbox[0]

                    margin = int(args.add_margin*(box[3] - box[2]))
                    merged_list.append(
                        [box[0]-margin, box[1]+margin, box[2]-margin, box[3]+margin])
    # may need to check if box is really in image
    return merged_list, free_list


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1,
                                                maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def get_image_list(horizontal_list, free_list, img, model_height=64):
    image_list = []
    maximum_y, maximum_x = img.shape[:2]

    max_ratio_hori, max_ratio_free = 1, 1
    for box in free_list:
        rect = np.array(box, dtype="float32")
        transformed_img = four_point_transform(img, rect)
        ratio = transformed_img.shape[1]/transformed_img.shape[0]
        crop_img = cv2.resize(transformed_img, (int(
            model_height*ratio), model_height), interpolation=Image.ANTIALIAS)
        # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        image_list.append((box, crop_img))
        max_ratio_free = max(ratio, max_ratio_free)

    max_ratio_free = math.ceil(max_ratio_free)

    for box in horizontal_list:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        crop_img = img[y_min: y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        ratio = width/height
        crop_img = cv2.resize(
            crop_img, (int(model_height*ratio), model_height), interpolation=Image.ANTIALIAS)
        image_list.append(
            ([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img))
        max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio)*model_height

    # sort by vertical position
    image_list = sorted(image_list, key=lambda item: item[0][0][1])
    return image_list, max_width


def text_dectection_postprocess(imgs, polys_list, args):
    image_lists = []
    for img, polys in zip(imgs, polys_list):
        result = easyocr_process(polys)
        imgH = 64
        horizontal_list, free_list = group_text_box(result, args)
        image_list, max_width = get_image_list(
            horizontal_list, free_list, img, model_height=imgH)
        image_lists.append(image_list)
    return image_lists


def main(yaml='args/detection_args.yaml',
         imgs=['image/example1.png',],
         device=2
         ):

    from imgcat import imgcat
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = load_ocr_map(yaml)
    args = args['model']['text_detection_args']
    net = load_detection_model(args, device)
    if args.refine:
        refined_net = load_detection_refined_model(args, device)
    else:
        refined_net = None

    ori_imgs = []
    for img in imgs:
        ori_imgs.append(cv2.imread(img))
    process_imgs = text_dectection_preprocess(ori_imgs)
    boxes_list, polys_list, ret_score_texts = do_detection(
        net, process_imgs, args, refined_net)
    bounds_list = text_dectection_postprocess(process_imgs, polys_list, args)

    # check
    for process_img, bounds, ret_score_text in zip(process_imgs, bounds_list, ret_score_texts):
        process_img = Image.fromarray(process_img)
        box_image = draw_boxes(process_img, bounds)
        imgcat(box_image)
        ret_score_text = cv2.cvtColor(ret_score_text, cv2.COLOR_BGR2RGB)
        imgcat(ret_score_text)


if __name__ == '__main__':
    fire.Fire(main)
