import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args.get("image"))
cv2.imshow("Original", image)






#Matrix of ones with dimensions of image, multiply by 100 --> matrix of one hundreds
# M = np.ones(image.shape, dtype = "uint8") * 100
# added = cv2.add(image, M)
# cv2.imshow("Added", added)
#
# M = np.ones(image.shape, dtype = "uint8") * 100
# sub = cv2.subtract(image, M)
# cv2.imshow("Subtracted", sub)

cv2.imshow("Added", imutils.changeBrightness(image, 50))
cv2.imshow("Subtracted", imutils.changeBrightness(image, -50))

cv2.waitKey(0)
