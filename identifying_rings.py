#Python 3.7.
import numpy as np
import argparse
import cv2
import imutils

"""
The Goal is, given images with a stack of 0, 1, or 4, orange rings, the program should be able
to detect how many rings are in the stack. I attempt to detect using multiple methods (___) and taking an average of the results.
"""

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
image = imutils.resize(image, newWidth = 720)   #Ensure the image is 720 width so the crop will work

image = imutils.crop(image, (260, 210), (460, 340))   #Crop image to remove background noise
cv2.imshow("Cropped", image)

#Turn all non-orange pixels black.
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print("width and height: " + str(imageHSV.shape[1]) + ", " + str(imageHSV.shape[0]))
for r in range(0, imageHSV.shape[0] - 1, 1):    #Loop through the width of the img (200 is width after crop)
    for c in range(0, imageHSV.shape[1] - 1, 1):    #Loop through the height of the img
        (hue, sat, val) = imageHSV[r, c]
        if hue <= 20 or hue >= 50:
            print("(%d, %d): %d" % (r, c, hue))
            image[r, c] = (0, 0, 0)
cv2.imshow("Rings", image)

#Turning into Greyscale and Blurring to prepare for edge detection and contouring
blurred = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(blurred, (15, 15), 0)
cv2.imshow("Blurred", blurred)

#adaptiveThreshold(image, max value, method, method, neighborhood area (k x k), constant to subtract from the mean)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)
cv2.imshow("Gaussian (Weighted Mean) Thresh", thresh)

edged = cv2.Canny(image, 30, 150)   #apply edge detection
cv2.imshow("Edged", edged)

cv2.waitKey(0)
