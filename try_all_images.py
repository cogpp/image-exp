__author__ = 'phillg07'

import numpy as np
import scipy.optimize as opt
import cv2
import json
import random
import facebreadcrums
import gabour
import glob
import os

img = None
all_scores = []
big = 1e20
score_template = None

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
        if not self.inbounds():
            print "oob!"
            return big
        subimg = img[int(self.y):(int(self.y+self.height)), int(self.x):(int(self.x+self.width))].copy()
        # subimg -= 128
        if subimg.size <= 0:
            print "empty image!"
            return big
        all_scores.append(self.weight * np.sum(subimg))
        return self.weight * np.sum(subimg)

    def inbounds(self):
        height, width = img.shape
        bottom_right_x, bottom_right_y = self.bottom_right()
        return 0 <= self.x <= width and 0 <= self.y <= height and bottom_right_x <= width and bottom_right_y <= height

    def transform(self, xs):
        height, width = img.shape
        return ScoreBox(x=(self.x + xs[0]*width), y=(self.y + xs[1]*height), width=self.width * xs[2], height=self.height * xs[2], weight=self.weight)

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

def objective(xs):
    new_boxes = transform_boxes(score_template, xs)
    if False:
        img_temp = img.copy()
        img_temp -= np.min(img_temp)
        if (np.min(img_temp) != np.min(img_temp)):
            img_temp /= np.max(img_temp)
        for box in new_boxes:
            box.draw(img_temp)
        cv2.imshow('image-temp', img_temp)
        cv2.waitKey(1)
    return score_boxes(new_boxes, img) #+ 0.1 + sci2.logit(1-xs[2])

def find_interesting_box(imname):
    gab = gabour.gabor_filter_image('resources/images-to-try/' + imname).astype(float)
    gab-=np.min(gab)
    gab/=np.max(gab)
    # cv2.imshow('gab', gab)
    # cv2.setMouseCallback('gab', get_intensity_callback, gab)
    # cv2.waitKey(0)

    face = facebreadcrums.get_breadcrumb_map_to_faces('resources/images-to-try/' + imname).astype(float)
    if (np.min(face) != np.max(face)):
        face -= np.min(face)
        face/=np.max(face)
    # cv2.imshow('face', face)
    # cv2.setMouseCallback('face', get_intensity_callback, face)
    # cv2.waitKey(0)

    global img
    img = cv2.multiply(gab, face)
    img-=np.min(img)
    img/=np.max(img)
    # cv2.imshow('toopt', img)
    # cv2.setMouseCallback('toopt', get_intensity_callback, img)
    # cv2.waitKey(0)

    global score_template
    score_template = load_template()

    xopt_best = []
    best_score = big
    i=0
    while i<100:
        x0 = np.array([random.random(), random.random(), random.random()])
        new_boxes = transform_boxes(score_template, x0)
        if not new_boxes[0].inbounds():
            continue
        xopt = opt.fmin(objective, x0)

        new_boxes = transform_boxes(score_template, xopt)
        score = score_boxes(new_boxes, img)
        if score < best_score:
            xopt_best = xopt
            best_score = score
        i+=1
    print "Result! = " + xopt.__str__()

    for score_boxe in transform_boxes(score_template, xopt_best):
        print "Transformed Box = " + score_boxe.__str__()
        score_boxe.draw(img)

    orig_image = cv2.imread('resources/images-to-try/' + imname);
    score_boxe.draw(orig_image)
    cv2.imwrite('result/' + imname, orig_image)

for file in os.listdir("resources/images-to-try"):
    print file
    find_interesting_box(file)