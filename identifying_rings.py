#Python 3.7.
import numpy as np
import argparse
import cv2
import imutils

"""
The Goal is, given images with a stack of 0, 1, or 4, orange rings, the program should be able
to detect how many rings are in the stack. I attempt to detect using multiple methods (___) and taking an average of the results.
"""
avgHueResult = -1
cntAspectRatioResults = -1


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())


"""
Note: The Photo Library I am using is very limited (very few teams have received the game elements),
so the camera is at different angles in all the trials, but is especially different comparing the
"no obstacles" images to the "w.obstacles" images. To demonstrate that my OpenCV algorithm which
detects the average hue works, I use a different crop based on the boolean 'obstacles'. In the real
robot game, the camera view is adjustable and should be much more consistent.
"""
obstacles = True
if "no obstacles" in args["image"]:
    obstacles = False


image = cv2.imread(args["image"])
image = imutils.resize(image, newWidth = 720)   #Ensure the image is 720 width so the crop will work
# cv2.imshow("Original", image)
original = image.copy()

image = imutils.crop(image, (300, 220), (450, 330))   #Crop image to remove background noise
cv2.imshow("Cropped", image)


"""
Detection Method 1: find the average hue in a cropped image of the stack, and compare them
to threshold values
"""
hueTotal = 0
imageHSVCropped = imutils.crop(original, (340, 270), (420, 320))   #Crop image to remove background noise
if obstacles:
    imageHSVCropped = imutils.crop(original, (320, 250), (400, 300))   #Crop image to remove background noise
for r in range(0, imageHSVCropped.shape[1] - 1):    #Loop through the width of the img
    for c in range(0, imageHSVCropped.shape[0] - 1):    #Loop through the height of the img
        (hue, sat, val) = imageHSVCropped[c, r]
        hueTotal = hueTotal + hue
hueAvg = hueTotal / (imageHSVCropped.shape[1] * imageHSVCropped.shape[0])
cv2.imshow("imageHSVCropped", imageHSVCropped)
print("avg Hue: %d" % hueAvg)
if hueAvg > 82:
    avgHueResult = 4
elif hueAvg > 72:
    avgHueResult = 1
else:
    avgHueResult = 0
print(avgHueResult)


"""
Detection Method 2: Isolate the ring stack, create a contour around it, then detect
the height to width ratio of the contour. This seems to be the most robust.
"""
#Turn all non-orange pixels black.
ringsOnlyImg = image.copy()
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
for r in range(0, imageHSV.shape[1] - 1):    #Loop through the width of the img
    for c in range(0, imageHSV.shape[0] - 1):    #Loop through the height of the img
        (hue, sat, val) = imageHSV[c, r]
        if hue < 10 or hue > 18 or sat < 80:
            ringsOnlyImg[c, r] = (0, 0, 0)
cv2.imshow("Rings", ringsOnlyImg)

#Turning into Greyscale and Blurring to prepare for edge detection and contouring
blurred = cv2.cvtColor(ringsOnlyImg, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
cv2.imshow("Blurred", blurred)

#adaptiveThreshold(image, max value, method, method, neighborhood area (k x k), constant to subtract from the mean)
# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)
# cv2.imshow("Gaussian (Weighted Mean) Thresh", thresh)

#apply edge detection
edged = cv2.Canny(blurred, 100, 400)   #2 threshold values have to be very high to ignore the gaps between rings, and irgnore other imperfections in the bg
cv2.imshow("Edged", edged)

#get the contours
#cnts is a list of the actual contours
(cnts, useless) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, cnts, -1, (255, 120, 0), 2)   #You can only draw one contour at a time, or all of them
cv2.imshow("Image with Contours", image)
cv2.waitKey(0)

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)  #returns tuple with rectangle dimensions to bound the contour c
    obj = image[y:y + h, x:x + w]   #cropping the image
    cv2.imshow("Object #{}".format(i), obj)

    if w > (2*h):
        cntAspectRatioResults = 1
    else:
        cntAspectRatioResults = 4

if len(cnts) == 0:
    cntAspectRatioResults = 0

print(cntAspectRatioResults)
print("Contours in the image: {}".format(len(cnts)))

cv2.waitKey(0)
