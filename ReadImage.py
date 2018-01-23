
#import cv2
import numpy as np
import pandas as pd
from scipy import misc

def processImage(image):
    pic = misc.imread(image)
    height = int(pic.shape[0]/2)
    halfimage = pic[0:height,:,:]
    return halfimage

def call_ep_imgdata(i):
    #There are 10 ep,
    imgset = []
    if i == 1:
        for j in range(1499): #1499
            path = './ev/deeptesla/RoadImage/ep%d/%d.jpg' % (i, j)
            halfimage = processImage(path)
            k = misc.imresize(halfimage, [66, 200, 3])
            imgset.append(k)

    elif i == 2:
        for j in range(3899):
            path = './ev/deeptesla/RoadImage/ep%d/%d.jpg' % (i, j)
            halfimage = processImage(path)
            k = misc.imresize(halfimage, [66, 200, 3])  #[66, 200, 3]
            imgset.append(k)

    else:
        for j in range(2699):
            path = './ev/deeptesla/RoadImage/ep%d/%d.jpg' % (i, j)
            halfimage = processImage(path)
            k = misc.imresize(halfimage, [66, 200, 3])
            imgset.append(k)
    return imgset

def steer_value(i):
    # i = 1~10
    path = './ev/deeptesla/epochs/ep%d.csv' % i
    value = pd.read_csv(path)
    leng = len(value) - 1
    number = value['wheel'][0:leng]
    return number

