#Python 3.7.
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred", image)

edged = cv2.Canny(image, 30, 150)   #apply edge detection
cv2.imshow("Edges", edged)

#get the contours
#cnts is a list of the actual contours
(cnts, useless) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("I count {} objects in this image".format(len(cnts)))

original = image.copy()
#parameters: image to draw on, the list of contours, which index in the list to draw (-1 --> draw all the contours in the list), color, thickness
cv2.drawContours(original, cnts, -1, (255, 120, 0), 2)   #You can only draw one contour at a time, or all of them
cv2.imshow("Image with Contours", original)
cv2.waitKey(0)


for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)  #returns tuple with rectangle dimensions to bound the contour c
    obj = image[y:y + h, x:x + w]   #cropping the image
    cv2.imshow("Object #{}".format(i), obj)

    mask = np.zeros(image.shape[:2], dtype = "uint8")   #full image array of zeros
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)    #returns tuple with circle dimensions of the smallest circle that encloses the contour
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius),255, -1) #draw the circle over the mask
    mask = mask[y:y + h, x:x + w]   #cropping again
    cv2.imshow("Masked Object", cv2.bitwise_and(obj, obj, mask =mask))
    cv2.waitKey(0)
