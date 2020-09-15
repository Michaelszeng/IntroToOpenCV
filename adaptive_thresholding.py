#Python 3.7.
import numpy as np
import argparse
import cv2

#Adaptive Threshes allow for multiple theshold values, which the program decides.

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred Greyscale", image)

#adaptiveThreshold(image, max value, method, method, neighborhood area (k x k), constant to subtract from the mean)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)

#adaptiveThreshold(image, max value, method, method, neighborhood area (k x k), constant to subtract from the mean)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian (Weighted Mean) Thresh", thresh)

cv2.waitKey(0)
