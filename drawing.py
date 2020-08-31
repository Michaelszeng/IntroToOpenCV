import numpy as np
import cv2

#zeros() makes numpy array of 0s, and datatype must be set to 8 bit integers bc RGB each have 255 = 2^8
canvas = np.zeros((500, 500, 3), dtype = "uint8")

black = (0, 0, 0)

#last parameter (-1) is thickness. -1 means solid
cv2.rectangle(canvas, (0, 0), (500, 500), black, -1)
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

cv2.line(canvas, (0, 0), (500, 500), (255, 255, 255))
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)

(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
blue = (255, 0, 0)
for r in range(0, 300, 50):
    #'np.random.randint' produce random int, adding parameter 'size=(3,)' makes it return a list of 3 random numbers
    cv2.circle(canvas, (centerX, centerY), r + np.random.randint(0, high=40), np.random.randint(0, high = 256, size = (3,)).tolist())
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
