__author__ = 'phillg07'

import numpy as np
import cv2
import random
import facebreadcrums

def build_filters():
    filters = []
    ksize = 127
    for theta in np.arange(0, np.pi, np.pi / 8):
        for lambd in np.arange(2, 16, 4):
            for psi in np.arange(0, np.pi/4, (np.pi/4) / 4):
                kern = cv2.getGaborKernel((ksize, ksize), 0.33*lambd, theta, lambd, 0.5, psi, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                # cv2.imshow('kern', kern / np.max(kern) * 255)
                # cv2.waitKey(0)
                filters.append(kern)
    return filters

def dog(img):
    inner = cv2.GaussianBlur(img, (129, 129), random.randint(0, 8))
    outer = cv2.GaussianBlur(img, (129, 129), random.randint(16, 64))
    dog = cv2.subtract(outer, inner)
    dog = dog.astype(np.float64)
    #dog = dog / np.max(dog) * 255
    dog += img * 0.25
    dog = dog / np.max(dog) * 128
    return dog.astype(np.uint8)


def repeatedly_dog(img):
    for i in range(0, 10):
        img = dog(img)
    return img


def winner(img):
    m = np.max(img)
    print m
    width, height = img.shape
    passes = []
    for x in range(0, width):
        for y in range(0, height):
            if img[x, y] == m:
                passes.append((x, y))
    return random.choice(passes)


def process(img, filters):
    accum = np.zeros_like(img).astype(np. float32)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        #fimg = repeatedly_dog(fimg)
        width, height =fimg.shape
        x, y = winner(fimg)
        fimg = facebreadcrums.create_breadcrumb_map([(x, y, 1, 1)], width, height)
        cv2.imshow('image', fimg)
        cv2.waitKey(1)
        #np.maximum(accum, fimg, accum)
        accum += fimg
    accum /= len(filters)
    return accum.astype(np.uint8)

img = cv2.imread('resources/tennis-b.jpg', 0)

#img_out = repeatedly_dog(img)


def get_intensity_callback(event,x,y,flags,image):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x)
        print(y)
        print(image[y,x])
        print("---")

filters = build_filters()
img_out = process(img, filters)
cv2.imwrite('resources/tennis-b-gab.jpg', img_out)
cv2.imshow('image', img_out)
cv2.setMouseCallback('image',get_intensity_callback, img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

