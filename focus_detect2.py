__author__ = 'phillg07'

import cv2
import scipy.ndimage.filters as flt
import numpy as np

img = cv2.imread('/Users/phillg07/Pictures/tennis2.jpg', 0)
img_res = cv2.subtract(img, cv2.GaussianBlur(img, (65, 65), 5))

cv2.imshow('image', img_res)
cv2.waitKey(0)
cv2.destroyAllWindows()