__author__ = 'phillg07'
import cv2
import scipy.ndimage.filters as flt
import numpy as np

img = cv2.imread('/Users/phillg07/Pictures/tennis2.jpg', 0)

max_l = np.percentile(img, 98)
min_l = np.percentile(img, 2)

ignore, img_max = cv2.threshold(flt.generic_filter(img, max, 10), max_l, 255, cv2.THRESH_BINARY)
ignore, img_min = cv2.threshold(flt.generic_filter(img, min, 10), min_l, 255, cv2.THRESH_BINARY_INV)
img_res = cv2.max(img_max, img_min)

cv2.imshow('image', img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
