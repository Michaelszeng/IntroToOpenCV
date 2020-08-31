import numpy as np
import cv2
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# r = 150.0 / image.shape[1]
# dim = (150, int(image.shape[0] * r))
#
# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("Resized (w = 150)", resized)

resized = imutils.resize(image, newHeight = 500)
cv2.imshow("Resized", resized)

cv2.waitKey(0)
