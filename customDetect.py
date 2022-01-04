import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

line1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
line2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
line3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm']

from models.common import DetectMultiBackend
from utils.general import (check_requirements, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

def formatImage(image):
    newImg = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    newImg = np.ascontiguousarray(newImg)
    return newImg

@torch.no_grad()
def detect(rawImage, show=False):
    weights = '1100_epoch.pt'  # model.pt path(s)
    data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
    conf_thres = 0.55  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    line_thickness = 2  # bounding box thickness (pixels)
    dnn = False  # use OpenCV DNN for ONNX inference
    image = formatImage(rawImage)
    # Load model
    device = select_device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    names= model.names
    # Run inference
    im = torch.from_numpy(image).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # Inference
    pred = model(im, augment=augment, visualize=False)
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process predictions
    det = pred[0]  # per image
    im0= rawImage.copy()
    keys = {}
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        # sort out multiple detections
        for key in det:
            classN = int(key[5])
            Bbox = key[0:4]
            conf = float(key[4])
            if names[classN] in keys:
                if keys[names[classN]]["Conf"] < conf:
                    keys[names[classN]] = {"Bbox": Bbox, "Conf": conf}
            else:
                keys[names[classN]] = {"Bbox": Bbox, "Conf": conf}
    #print out the result and draw bounding boxes
    if show:
        annotator = Annotator(rawImage.copy(), line_width=line_thickness, example=str(names))
        for key in keys:
            conf = keys[key]["Conf"]
            Bbox = keys[key]["Bbox"]
            print("Detected " + key + " with confidence: "+str(conf))
            #label = f'{key} {conf:.2f}'
            label = key
            annotator.box_label(Bbox, label, color=colors(names.index(key), True))

        im0 = annotator.result()
        cv2.imshow("Image with detected boys", im0)
        cv2.waitKey()
    return keys

def getPointsFromBboxes(keys):
    pointKeys = {}
    for key in keys:
        box = keys[key]["Bbox"]
        x1 = float(box[0])
        y1 = float(box[1])
        x2 = float(box[2])
        y2 = float(box[3])
        dx = round(x1+(x2-x1)/2)
        dy = round(y1+(y2-y1)/2)
        pointKeys[key] = np.array([dx, dy])
    return pointKeys

def showKeys(image, keys):
    for key in keys:
        image =cv2.circle(image, keys[key], radius=5, color=(0,0,255), thickness=4)
    cv2.imshow("Key points", image)
    cv2.waitKey()

def fixKeys(detectedPts):
    keys = fixLine(detectedPts, line1)
    keys = fixStart(keys, line1)
    keys = fixEnd(keys, line1)
    keys = fixLine(keys, line2)
    keys = fixStart(keys, line2)
    keys = fixLine(keys, line3)
    keys = fixStart(keys, line3)
    return keys

def fixStart(detectedPts, line):
    keys = detectedPts.copy()
    missed = []
    startP = 0
    for k in range(len(line)):
        if line[k] not in detectedPts:
            missed.append(line[k])
        else:
            startP = detectedPts[line[k]][0]
            break
    amount = len(missed)
    if amount < len(line):
        height = round(getAvgH(detectedPts, line))
        dist = round(getAvgVertD(detectedPts, line))
        for i in range(len(missed)):
            key = missed[amount-1-i]
            newX = startP-(i+1)*dist
            keys[key] = np.array([newX, height])
    return keys

def fixEnd(detectedPts, line):
    keys = detectedPts.copy()
    missed = []
    startP = 0
    for k in range(len(line)-1, -1, -1):
        if line[k] not in detectedPts:
            missed.append(line[k])
        else:
            startP = detectedPts[line[k]][0]
            break
    amount = len(missed)
    if amount < len(line):
        height = round(getAvgH(detectedPts, line))
        dist = round(getAvgVertD(detectedPts, line))
        for i in range(len(missed)):
            key = missed[amount - 1 - i]
            newX = startP + (i + 1) * dist
            keys[key] = np.array([newX, height])
    return keys

def fixLine(detectedPts, line):
    startId = -1
    for k in range(len(line)):
        if line[k] in detectedPts:
            startId = k
            break
    start = line[startId]
    miss = []
    keys = detectedPts.copy()
    for i in range(startId, len(line)):
        key = line[i]
        if key in detectedPts:
            if len(miss) != 0:
                sP = detectedPts[start]
                eP = detectedPts[key]
                pts = fixSpace(sP, eP, len(miss))
                for i in range(len(miss)):
                    keys[miss[i]] = pts[i]
                start = key
                miss = []
            else:
                start = key
        else:
            miss.append(key)
    return keys
def fixSpace(sP, eP, space):
    midpoints = []
    diff = eP-sP
    dist = np.linalg.norm(diff)
    direc = diff/dist
    step =dist/(space+1)
    for i in range(1, space+1):
        pt = np.round(sP+i*step*direc).astype(int)
        midpoints.append(pt)
    return midpoints

def cleanDetection(detectedPts):
    checked = {}
    avgH1, avgH2, avgH3 = getAvgHs(detectedPts)
    for key in detectedPts:
        hor = checkHorizontal(key, detectedPts)
        ver = checkVertical(key, detectedPts, avgH1, avgH2, avgH3)
        if hor and ver:
            checked[key] = detectedPts[key]
    return checked
def checkVertical(key, detectedPts, avgH1, avgH2, avgH3):
    check = False
    height = detectedPts[key][1]
    if key in line1:
        check = height > avgH2 and height > avgH3
    elif key in line2:
        check = height > avgH3 and height < avgH1
    elif key in line3:
        check = height < avgH1 and height < avgH2
    return check
def checkHorizontal(key, detectedPts):
    return True
def getAvgHs(detectedPts):
    avgH1 = getAvgH(detectedPts, line1)
    avgH2 = getAvgH(detectedPts, line2)
    avgH3 = getAvgH(detectedPts, line3)
    return avgH1, avgH2, avgH3

def getAvgH(detectedPts, line):
    missed = 0
    sum = 0
    for key in line:
        if key in detectedPts:
            sum += detectedPts[key][1]
        else:
            missed += 1
    avgH = sum / (len(line) - missed)
    return avgH

def getAvgVertD(detectedPts, line):
    startId = -1
    for k in range(len(line)):
        if line[k] in detectedPts:
            startId = k
            break
    start = line[startId]
    sum = 0
    tot = 0
    for i in range(startId+1, len(line)):
        if line[i] in detectedPts:
            sP = detectedPts[start][0]
            eP = detectedPts[line[i]][0]
            sum+=eP-sP
            tot+=1
            start = line[i]
        else:
            break
    return sum/tot

if __name__ == "__main__":
    check_requirements(exclude=('tensorboard', 'thop'))
    path = 'TestImages/keybTest8.png'
    image = cv2.imread(path)
    keys = detect(image, True)
    keys = getPointsFromBboxes(keys)
    showKeys(image, keys)
    keys = fixKeys(keys)
    showKeys(image, keys)
    print("End")
