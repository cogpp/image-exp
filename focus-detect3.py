__author__ = 'phillg07'

import cv2
import numpy as np

img = cv2.imread('/Users/phillg07/Pictures/tennis2.jpg', 0).astype(float)

mu = cv2.GaussianBlur(img, (41, 41), 20)
mu_2 = cv2.multiply(mu, mu)
mu_2g = cv2.GaussianBlur(mu_2, (41, 41), 20)
mu_sub = cv2.subtract(mu_2g, mu_2)
sigma = cv2.sqrt(mu_sub)

cv2.imshow('image', sigma)
cv2.waitKey(0)
cv2.destroyAllWindows()
