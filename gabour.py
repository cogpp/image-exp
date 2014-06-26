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
            for psi in [-np.pi/2, np.pi/2]:
                kern = cv2.getGaborKernel((ksize, ksize), 0.33*lambd, theta, lambd, 1, psi, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img).astype(np. float32)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_64F, kern)
        np.maximum(accum, fimg, accum)
    return accum


def gabor_filter_image(img_loc):
    img = cv2.imread(img_loc, 0)
    filters = build_filters()
    return process(img, filters)