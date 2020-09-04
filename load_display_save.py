import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args.get("image"))
print(image)
print("width: {} px".format(image.shape[1]))
print("height: {} px".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))

imageList = args.get("image").split('\\')
print(imageList)
imageName = imageList[-1]
print(imageName)
imageNameRaw = imageName[0:-4]
print(imageNameRaw)

cv2.imshow("Image", image)
cv2.waitKey(0)


cv2.imwrite(imageNameRaw + "new.jpg", image)
