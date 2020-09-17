#Python 3.7.
import numpy as np
import argparse
import cv2
import mahotas

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred Greyscale", image)


rows,cols = image.shape
for i in range(0, rows, 5):
  for j in range(0, cols, 5):
     k = image[i,j]
     print(k)


T = mahotas.thresholding.otsu(blurred)  #Getting the threshold value
print("Otsu's Threshold: {}".format(T))

thresh = image.copy()   #making a copy of the image to apply T to
thresh[thresh < T] = 0    #pixels less than thresh go to black
thresh[thresh > T] = 255    #pixels greater than thresh go to white
thresh = cv2.bitwise_not(thresh)    #effectively inverts colors, same thing is a binary invert
cv2.imshow("Otsu", thresh)



T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard Theshold: {}".format(T))

thresh = image.copy()
thresh[thresh < T] = 0
thresh[thresh > T] = 255
# thresh = cv2.bitwise_not(thresh)    #effectively inverts colors, same thing is a binary invert
cv2.imshow("RC", thresh)
cv2.waitKey(0)
