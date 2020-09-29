#Python 3.7
# numpy 1.15.4
import numpy as np  #Numeric/array calculations
import argparse     #Import arguments (the path to the image)
import cv2          #All the Computer Vision functions
import imutils      #My own class with simplified OpenCV functions
import math         #Use math functions

def main():
    """
    The Goal is, given images with a stack of 0, 1, or 4, orange rings, the program should be able
    to detect how many rings are in the stack. I attempt to detect using multiple methods, the results of which are printed at the end.
    """
    #Initalize the variables to hold the results
    avgHueResult = -1
    cntAspectRatioResults = -1
    houghRotationAspectRatioResults = -1

    #Get the path to the image of the rings
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


    image = cv2.imread(args["image"])   #Get the image from the argument
    image = imutils.resize(image, newWidth = 720)   #Ensure the image is 720 width so the crop will work
    # cv2.imshow("Original", image)
    original = image.copy()

    image = imutils.crop(image, (300, 220), (450, 330))   #Crop image to remove unecessary background noise
    cv2.imshow("Original Image Cropped", image)



    """
    Detection Method 1: Isolate the ring stack, create a contour around it, then detect
    the height to width ratio of the contour to determine the number of rings
    """
    edged = getEdges(image)     #Get image with black background and edges
    cv2.imshow("Method 1: Image with background removed", edged)

    #get the contours
    #The Parameters are tuned such a maximum of 1 contour is returned for 1 or 4 rings, and 0 contours are returned for 0 rings
    (cnts, useless) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    #cnts is a list of the actual contours
    cv2.drawContours(image, cnts, -1, (255, 120, 0), 2)   #Drawing the contour
    cv2.imshow("Method 1: Contour Around Rings", image)

    #If there are no contours, there are 0 rings
    if len(cnts) == 0:
        cntAspectRatioResults = 0
    else:
        (x, y, w, h) = cv2.boundingRect(cnts[0])  #returns tuple with rectangle dimensions to bound the contour c
        obj = image[y:y + h, x:x + w]   #cropping the image

        #Based on the ration of width to height of the contour of the rings, determine the height of the stack
        if w > (2*h):
            cntAspectRatioResults = 1
        else:
            cntAspectRatioResults = 4


    print("# Rings Based on Width and Height of Contour: %d" % cntAspectRatioResults)
    # print("Contours in the image: {}".format(len(cnts)))
    cv2.waitKey(0)



    """
    Detection Method 2: find the average hue in a cropped image of the stack, and compare them
    to threshold values to determine the ring stack
    """
    imageCropped = imutils.crop(original, (340, 270), (420, 320))   #Crop image to remove background noise
    if obstacles:
        imageCropped = imutils.crop(original, (320, 250), (400, 300))   #Crop image to remove background noise
    cv2.imshow("Method 2: Cropped Image", imageCropped)

    imageCroppedIsolated = removeBackground(imageCropped)   #color the background black to remove noise
    cv2.imshow("Method 2: Image with background removed", imageCroppedIsolated)
    imageHSVCroppedIsolated = cv2.cvtColor(imageCroppedIsolated, cv2.COLOR_BGR2HSV) #Converting to HSV
    cv2.imshow("Method 2: Image, cropped, and in HSV", imageHSVCroppedIsolated)

    hueTotal = 0
    for r in range(0, imageHSVCroppedIsolated.shape[1] - 1):    #Loop through the width of the img
        for c in range(0, imageHSVCroppedIsolated.shape[0] - 1):    #Loop through the height of the img
            (hue, sat, val) = imageHSVCroppedIsolated[c, r]     #Get HSV values of the pixel
            hueTotal = hueTotal + hue   #Append to the total hue value
    hueAvg = hueTotal / (imageHSVCroppedIsolated.shape[1] * imageHSVCroppedIsolated.shape[0])   #Calculate the average hue
    print("avg Hue: %d" % hueAvg)
    #Based on the hue, determine the heigh of the ring stack based on pre-deterined thesholds
    if hueAvg >= 4:
        avgHueResult = 4
    elif hueAvg >= 1:
        avgHueResult = 1
    else:
        avgHueResult = 0
    print("# Rings Based on Average Hue on HSV, Cropped, and Removed Background: %d" % avgHueResult)
    cv2.waitKey(0)



    """
    Detection Method 3: Finding the contour around the ring stack, drawing it over a
    black background, then using Hough Line Detection to find the angle of the camera,
    then rotating the image based on the camera's tilt. Finally, compare the new height
    and width ratio of the contour and determine the number of rings.
    """
    edged = getEdges(image)     #Finding edges
    edgedNoLines = edged.copy() #Making a copy of the edged image for later use
    edged = cv2.GaussianBlur(edged, (5, 5), 0)  #Blurring image for Improved Line Detection
    cv2.imshow("Method 3: Image edges: ", edged)
    #Last Parameter is Threshold, most be greater than threshold to be counted
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, 60)    #Detecting the lines
    # print(lines)

    try:    #If there are no lines detected, the below code will throw an exception, so I add a catch block
        horizSlopesSum = 0.0
        numHorizSlopes = 0
        for line in lines:  #Draw the lines found
            x1, y1, x2, y2 = line[0]    #Get endpoints of line
            cv2.line(edged, (x1, y1), (x2, y2), (255, 255, 255), 2)  #Draw Line
            if x2 - x1 != 0:    #Ensure we're not dividing by zero
                slope = (y2-y1) / (x2-x1)
                if abs(0.0 - slope) < 0.25:     #If the slope is less than 0.25, then we consider it a line that's supposed to be horizontal
                    horizSlopesSum += slope
                    numHorizSlopes += 1
        avgHorizSlope = horizSlopesSum / numHorizSlopes     #Finding the avg slope of the lines that are supposed to be horizontal
        # print("Slope: %f" % avgHorizSlope)
        angle = math.degrees(math.atan(avgHorizSlope))  #Convert slope to an angle for rotation
        # print("angle: %f" % angle)
        cv2.imshow("Method 3: Lines Detected from Hough Line Detection", edged)

        rotated = imutils.rotate(edgedNoLines, angle)   #Rotating the Image
        cv2.imshow("Method 3: Edges rotated based on Angle of Lines", rotated)

        #get the contours
            #The Parameters are tuned such a maximum of 1 contour is returned for 1 or 4 rings, and 0 contours are returned for 0 rings
        (cnts, useless) = cv2.findContours(rotated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    #cnts is a list of the actual contours
        cv2.drawContours(image, cnts, -1, (255, 120, 0), 2)   #Drawing the contour
        cv2.imshow("Method 3: Contour Around Rotated Image", image)

        #If there are no contours, there are 0 rings
        if len(cnts) == 0:
            houghRotationAspectRatioResults = 0
        else:
            (x, y, w, h) = cv2.boundingRect(cnts[0])  #returns tuple with rectangle dimensions to bound the contour c
            obj = image[y:y + h, x:x + w]   #cropping the image

            #Based on the ration of width to height of the contour of the rings, determine the height of the stack
            if w > (2*h):
                houghRotationAspectRatioResults = 1
            else:
                houghRotationAspectRatioResults = 4

    except:     #Catch in case 'lines' is None (no lines detected)
        print("Exception Thrown")
        houghRotationAspectRatioResults = 0

    print("# Rings based on rotated image and detecting aspect ratio: %d" % houghRotationAspectRatioResults)
    cv2.waitKey(0)


def removeBackground(image):
    """
    Purpose: Turns all non-orange pixels to black, removes all the noise to improve
    accuracy and consistency.
    Parameters: The image to manipulate
    Returns: The image with the background turned black
    """
    ringsOnlyImg = image.copy()     #Don't want to edit the original image
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   #Convert to HSV to do hue detection
    for r in range(0, imageHSV.shape[1] - 1):    #Loop through the width of the img
        for c in range(0, imageHSV.shape[0] - 1):    #Loop through the height of the img
            (hue, sat, val) = imageHSV[c, r]    #Get the HSV values of current pixel
            if hue < 10 or hue > 18 or sat < 80:    #If the hue value is not orange:
                ringsOnlyImg[c, r] = (0, 0, 0)      #Set it to black
    return ringsOnlyImg     #Return image with black background

def getEdges(image):
    """
    Purpose: return an image with a removed background and with edges detected.
    Parameters: the image to manipulate
    Returns: the image without the background and with the edges
    """
    #Turn all non-orange pixels black.
    ringsOnlyImg = removeBackground(image)
    # cv2.imshow("Rings", ringsOnlyImg)

    #Turning into Greyscale and Blurring to prepare for edge detection and contouring
    blurred = cv2.cvtColor(ringsOnlyImg, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    # cv2.imshow("Blurred", blurred)

    #apply edge detection
    edged = cv2.Canny(blurred, 100, 400)   #2 threshold values have to be very high to ignore the gaps between rings, and irgnore other imperfections in the bg
    # cv2.imshow("Edged", edged)
    return edged



if __name__ == "__main__":  #Run the main function
    main()
