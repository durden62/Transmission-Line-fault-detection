import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from flask import Flask, jsonify, request

def detect(source, save_img=False):
    weights= 'bestofbest.pt'
    imgsz = 640

    set_logging()
    device = select_device('')
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    check_img_size(imgsz, s = model.stride.max())

    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0,255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1,3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    img = torch.from_numpy(source).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.dimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)


    for i, det in enumerate(pred):

