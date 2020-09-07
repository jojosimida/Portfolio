import os
from collections import OrderedDict

import cv2
import fire
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from text_detection.CRAFT.craft import CRAFT
from text_detection.CRAFT.craft_utils import (adjustResultCoordinates,
                                              getDetBoxes)
from text_detection.CRAFT.imgproc import (cvt2HeatmapImg,
                                          normalizeMeanVariance,
                                          resize_aspect_ratio)
from text_detection.pipeline.preprocessing import text_dectection_preprocess
from text_detection.pipeline.utils import load_ocr_map


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(net, images, args, refined_net=None):
    x = []
    ratio_hs = []
    device = next(net.parameters()).device
    for image in images:
        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, args.canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        _x = normalizeMeanVariance(img_resized)
        _x = torch.from_numpy(_x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        _x = Variable(_x).to(device) if args.cuda else Variable(
            _x)  # [c, h, w] to [b, c, h, w]

        x.append(_x)
        ratio_hs.append(ratio_h)

    x = torch.stack(x)
    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_texts = y[:, :, :, 0].cpu().data.numpy()
    score_links = y[:, :, :, 1].cpu().data.numpy()

    # refine link
    if refined_net:
        print('Do refined model')
        with torch.no_grad():
            y_refiner = refined_net(y, feature)
        score_links = y_refiner[:, :, :, 0].cpu().data.numpy()

    boxes_list = []
    polys_list = []
    ret_score_texts = []
    for score_text, score_link in zip(score_texts, score_links):
        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, args.text_threshold, args.link_threshold, args.low_text, args.poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img)

        boxes_list.append(boxes)
        polys_list.append(polys)
        ret_score_texts.append(ret_score_text)
    return boxes_list, polys_list, ret_score_texts


def load_detection_model(args, device='cpu'):
    net = CRAFT()

    print('Loading weights from checkpoint (' + args.trained_model + ')')

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=device)))
        cudnn.benchmark = False
    else:
        net.load_state_dict(copyStateDict(torch.load(
            args.trained_model, map_location='cpu')))

    net.eval()
    return net


def load_detection_refined_model(args, device='cpu'):
    from text_detection.CRAFT.refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(
            copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.to(device)
    else:
        refine_net.load_state_dict(copyStateDict(
            torch.load(args.refiner_model, map_location='cpu')))
    refine_net.eval()
    args.poly = True
    return refine_net


def do_detection(net, images, args, refined_net):
    bboxes, polys, ret_score_text = test_net(net, images, args, refined_net)
    return bboxes, polys, ret_score_text


def main(yaml='args/detection_args.yaml',
         imgs=['image/TA01_24.tif', ],
         ):

    os.environ["CUDA_VISIBLE_DEVICES"]="2" 
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

    from imgcat import imgcat
    for process_img, ret_score_text in zip(process_imgs, ret_score_texts):
        imgcat(process_img)
        ret_score_text = cv2.cvtColor(ret_score_text, cv2.COLOR_BGR2RGB)
        imgcat(ret_score_text)

    import pdb;pdb.set_trace()

if __name__ == '__main__':
    fire.Fire(main)
