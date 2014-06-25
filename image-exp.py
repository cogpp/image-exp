__author__ = 'phillg07'

import numpy as np
import scipy.special as sci2
import scipy.optimize as opt
import cv2
import json
import math
import sys
import random

big = 1e20
all_scores = []

class ScoreBox:
    """This represents a box for weighting the score an image transform
    """
    def __init__(self, from_dict=None,  x=0, y=0, width=0, height=0, weight=0):
        if from_dict is not None:
            self.__dict__ = from_dict
        else:
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.weight = weight

    def __str__(self):
        return json.dumps(self.__dict__)

    def color(self):
        return (255, 255, 255)


    def offset_pix(self):
        return int(self.x), int(self.y)

    def size_pix(self):
        return int(self.width), int(self.height)

    def bottom_right(self):
        return int(self.x + self.width), int(self.y + self.height)

    def draw(self, img):
        cv2.rectangle(img, self.offset_pix(), self.bottom_right(), self.color(), 2)


    def score(self, img):
        if not self.inbounds((self.x, self.y)) or not self.inbounds(self.bottom_right()):
            print "oob!"
            return big
        subimg = img[int(self.y):(int(self.y+self.height)), int(self.x):(int(self.x+self.width))].copy().astype(np.float32)
        subimg -= 128
        if subimg.size <= 0:
            print "empty image!"
            return big
        all_scores.append(self.weight * np.sum(subimg))
        return self.weight * np.sum(subimg)

    def inbounds(self, point):
        return 0 <= point[0] <= 1280 and 0 <= point[1] <= 720

    def transform(self, xs):
        return ScoreBox(x=(self.x + xs[0]*1280), y=(self.y + xs[1]*720), width=self.width * xs[2], height=self.height * xs[2], weight=self.weight)


def load_template():
    with open("resources/thumbnail_template.json", "r") as template_json:
        json_file = json.loads(template_json.read())
        return [ScoreBox(i) for i in json_file["scoreBoxes"]]

def score_boxes(sboxes, img):
    s = sum([box.score(img) for box in sboxes])
    print "s=" + s.__str__()
    return s

def transform_boxes(boxes, transform):
    print transform
    return [box.transform(transform) for box in boxes]

def get_intensity_callback(event,x,y,flags,image):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x)
        print(y)
        print(image[y,x])
        print("---")

def get_intensity_callback(event,x,y,flags,image):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x)
        print(y)
        print(image[y,x])
        print("---")

gab = cv2.imread('resources/tennis-b-gab.jpg', 0).astype(float)
face = cv2.imread('resources/facebreadcrumb.jpg', 0).astype(float)

face *= face

img = cv2.multiply(gab, face)
img = img / np.max(img) * 255
img = img.astype(np.uint8)


score_template = load_template()



debug = False

def objective(xs):
    new_boxes = transform_boxes(score_template, xs)
    if debug:
        img_temp = img.copy()
        for box in new_boxes:
            box.draw(img_temp)
        cv2.imshow('image-temp', img_temp)
        cv2.waitKey(1)
    return score_boxes(new_boxes, img) #+ 0.1 + sci2.logit(1-xs[2])


xopt_best = []
best_score = big
for i in range(1, 100):
    x0 = np.array([random.random(), random.random(), random.random()])
    #xopt, ignore = opt.anneal(objective, x0, lower=0.0, upper=1.0)
    xopt = opt.fmin(objective, x0)
    new_boxes = transform_boxes(score_template, xopt)
    score = score_boxes(new_boxes, img)
    if score < best_score:
        xopt_best = xopt
        best_score = score
    # xopt_map = opt.basinhopping(objective, x0, stepsize=0.05)
    # xopt = xopt_map["x"]
    #xopt = opt.brute(objective, [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])

print "Result! = " + xopt.__str__()



for score_boxe in transform_boxes(score_template, xopt_best):
    print "Transformed Box = " + score_boxe.__str__()
    print score_boxe.score(img)
    score_boxe.draw(img)

cv2.imshow('image', img)
cv2.setMouseCallback('image', get_intensity_callback, img)
cv2.waitKey(0)
cv2.destroyAllWindows()