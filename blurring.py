#Python 3.7.3
#matplotlib 3.3.1
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

#Stack all images onto a single window in a row (there is no vstack[])
blurred = np.hstack([cv2.blur(image, (3, 3)), cv2.blur(image, (5, 5)), cv2.blur(image, (7, 7))])
cv2.imshow("Averaged", blurred) #Center pixel is avg of the k x k grid around it
cv2.waitKey(0)

# blur3 = cv2.blur(image, (3, 3))
# blur5 = cv2.blur(image, (5, 5))
# blur7 = cv2.blur(image, (7, 7))
# cv2.imshow("3", blur3)
# cv2.imshow("5", blur5)
# cv2.imshow("7", blur7)
# cv2.waitKey(0)

blurred = np.hstack([cv2.GaussianBlur(image, (3, 3), 0), cv2.GaussianBlur(image, (5, 5), 0), cv2.GaussianBlur(image, (7, 7), 0)])
cv2.imshow("Gaussian", blurred) #Weighted avg based on distance from center
cv2.waitKey(0)

blurred = np.hstack([cv2.medianBlur(image, 3), cv2.medianBlur(image, 5), cv2.medianBlur(image, 7)])
cv2.imshow("Median", blurred)   #Median based on surrounding k x k grid--best at eliminating salt/pepper noise
cv2.waitKey(0)

blurred = np.hstack([cv2.bilateralFilter(image, 5, 21, 21), cv2.bilateralFilter(image, 7, 31, 31), cv2.bilateralFilter(image, 9, 41, 41)])
cv2.imshow("Bilateral", blurred)    #Weighted avg, only considering pixels in a k x k grid with similar intensity already. Maintains edges, but reduces noise
cv2.waitKey(0)
