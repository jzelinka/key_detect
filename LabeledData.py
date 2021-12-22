import torch
import json

import torchvision
from PIL import Image
from torchvision import transforms as tf
from torchvision.io import read_image
import numpy as np

def getLabelNum(label):
    if label == 'a':
        return 0
    elif label == 'b':
        return 1
    elif label == 'c':
        return 2
    elif label == 'd':
        return 3
    elif label == 'e':
        return 4
    elif label == 'f':
        return 5
    elif label == 'g':
        return 6
    elif label == 'h':
        return 7
    elif label == 'i':
        return 8
    elif label == 'j':
        return 9
    elif label == 'k':
        return 10
    elif label == 'l':
        return 11
    elif label == 'm':
        return 12
    elif label == 'n':
        return 13
    elif label == 'o':
        return 14
    elif label == 'p':
        return 15
    elif label == 'q':
        return 16
    elif label == 'r':
        return 17
    elif label == 's':
        return 18
    elif label == 't':
        return 19
    elif label == 'u':
        return 20
    elif label == 'v':
        return 21
    elif label == 'w':
        return 22
    elif label == 'x':
        return 23
    elif label == 'y':
        return 24
    elif label == 'z':
        return 25
    elif label == 'shift':
        return 26
    elif label == 'num':
        return 27
    elif label == 'space':
        return 28
    elif label == 'enter':
        return 29
    elif label == 'del':
        return 30
    else:
        print("Unexpected key class name")
        return 666

def numsToLables(lables):
    names = []
    for label in lables:
        names.append(numToLab(label))
    return names
def numToLab(num):
    if num == 0:
        return 'a'
    elif num == 1:
        return 'b'
    elif num == 2:
        return 'c'
    elif num == 3:
        return 'd'
    elif num == 4:
        return 'e'
    elif num == 5:
        return 'f'
    elif num == 6:
        return 'g'
    elif num == 7:
        return 'h'
    elif num == 8:
        return 'i'
    elif num == 9:
        return 'j'
    elif num == 10:
        return 'k'
    elif num == 11:
        return 'l'
    elif num == 12:
        return 'm'
    elif num == 13:
        return 'n'
    elif num == 14:
        return 'o'
    elif num == 15:
        return 'p'
    elif num == 16:
        return 'q'
    elif num == 17:
        return 'r'
    elif num == 18:
        return 's'
    elif num == 19:
        return 't'
    elif num == 20:
        return 'u'
    elif num == 21:
        return 'v'
    elif num == 22:
        return 'w'
    elif num == 23:
        return 'x'
    elif num == 24:
        return 'y'
    elif num == 25:
        return 'z'
    elif num == 26:
        return 'shift'
    elif num == 27:
        return 'num'
    elif num == 28:
        return 'space'
    elif num == 29:
        return 'enter'
    elif num == 30:
        return 'del'
    else:
        print("Unexpected key class number!")
        return '#'

def loadData(datType):
    numOfData = 102
    boxes = []
    images = []
    labels = []
    for i in range(numOfData+1):
        if i == 4 or i==37 or i == 44 or i == 47 or i == 49 or i == 50 or i==51 or i==52 or i==54 or i == 58:
            continue
        if i == 59 or i == 60 or i == 87 or i == 88:
            continue
        pathLabel = 'data\labels\keyb'+ str(i) + '.json'
        pathImage = 'data\keyb' + str(i) + '.png'
        im = read_image(pathImage)
        images.append(im)

        f = open(pathLabel)
        data = json.load(f)
        # anno = {}
        # anno['filename'] = 'keyb'+ str(i) + '.png'
        # anno['width'] = data['imageWidth']
        # anno['height'] = data['imageWidth']
        label = []
        points = []
        for shape in data['shapes']:
            label.append(getLabelNum(shape['label']))
            point = shape['points']
            xmin = min(point[0][0], point[1][0])
            xmax = max(point[0][0], point[1][0])
            ymin = min(point[0][1], point[1][1])
            ymax = max(point[0][1], point[1][1])
            points.append([xmin, ymin, xmax, ymax])

        labels.append(label)
        boxes.append(torch.tensor(points))
    dataset = {}
    dataset['data'] = images
    dataset['boxes'] = boxes
    dataset['labels'] = labels
    return dataset

class Dataset():
    def __init__(self, type):
        raw = loadData(type)
        self.data = raw['data']
        self.labels = raw['labels']
        self.boxes = raw['boxes']
        # bounding = torchvision.utils.draw_bounding_boxes(self.data[50], self.boxes[50], numsToLables(self.labels[50]), colors="red")
        # transf = tf.ToPILImage()
        # pilmage = transf(bounding)
        # pilmage.show()

    def __getitem__(self, item):
        batch = {'data': self.data[item].type(torch.float32).permute(2,0,1),
                 'labels': self.labels[item],
                 'boxes':self.boxes[item]}
        return batch

    def __len__(self):
        return len(self.labels)

def getDataLoader(batchSize, shuffler):
    data = Dataset(0)
    dataLoader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=shuffler)
    return dataLoader

if __name__ == '__main__':
    loader = getDataLoader(5, True)
    print("Working on it!")