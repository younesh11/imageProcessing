import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("test1.png")
#  on the copied image
copyimg = image.copy()
h, w = copyimg.shape[:2]
# creating a mask
mask = np.zeros([h + 2, w + 2], np.uint8)
#flooding wrt points  (50, 20, 50), (50, 50 ,30)
img1 = cv2.floodFill(copyimg, mask, (1631, 1430), (0, 255, 0), (50, 20, 50), (50, 50 ,30), cv2.FLOODFILL_FIXED_RANGE)
lower_green = np.array([0, 255, 0])
upper_green = np.array([0, 255, 0])
mask = cv2.inRange(copyimg, lower_green, upper_green)
res = cv2.bitwise_and(copyimg,copyimg, mask = mask)

cv2.imwrite("newimg.png", res)
print("image saved")
nimg = cv2.imread("newimg.png", 0)
contours, hierarchy = cv2.findContours(nimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image, contours, -1, (255, 0, 0), 4)
plt.imshow(image)
plt.show()
print(contours)
import pickle
a = contours
serialized = pickle.dumps(a, protocol=0)
print(serialized)



