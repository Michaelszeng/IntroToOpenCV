import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args.get("image"))
cv2.imshow("Ori")
