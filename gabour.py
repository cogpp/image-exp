__author__ = 'phillg07'

import numpy as np
import cv2

def build_filters():
    filters = []
    ksize = 127
    for theta in np.arange(0, np.pi, np.pi / 16):
        for lambd in np.arange(4, 16, 2):
            for  psi in np.arange(0, np.pi/4, (np.pi/4) / 4):
                kern = cv2.getGaborKernel((ksize, ksize), 0.33*lambd, theta, lambd, 0.5, psi, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                # cv2.imshow('kern', kern / np.max(kern) * 255)
                # cv2.waitKey(0)
                filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        cv2.imshow('image', fimg)
        cv2.waitKey(1)
        np.maximum(accum, fimg, accum)
    return accum

img = cv2.imread('resources/tennis-b.jpg', 0)

filters = build_filters()
img_out = process(img, filters)
cv2.imwrite('resources/tennis-b-gab.jpg', img_out)
cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()