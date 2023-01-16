import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('1-70.tiff', 0)

ret, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
cc = cv.connectedComponentsWithStats(thresh, 4, cv.CV_32S)
numLabels = cc[0]
labels = cc[1]
stats = cc[2]
centroids = cc[3]

output = np.zeros(img.shape, dtype="uint8")

for i in range(numLabels):
    area = stats[i, cv.CC_STAT_AREA]
    if (area > 100) and (area < 500):
        componentMask = (labels == i).astype("uint8") * 255
        output = cv.bitwise_or(output, componentMask)

#plt.imshow(thresh, 'gray')
#plt.show()
#plt.imshow(output, 'gray')
#plt.show()
cv.imshow('threshold', thresh)
cv.imshow('output', output)
cv.waitKey(0)

cv.imwrite('components.tiff', output)
