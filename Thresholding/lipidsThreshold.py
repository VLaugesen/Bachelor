import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = cv.imread('D:\Bachelor Project\Bachelor\Data\s2720_e01.tif', 0)
x,y,h,w = 880, 780, 5000, 5000 
croppedImg = img[y:y+h, x:x+w]



ret, thresh = cv.threshold(croppedImg, 240, 255, cv.THRESH_BINARY)
cc = cv.connectedComponentsWithStats(thresh, 8, cv.CV_32S)

numLabels = cc[0]
labels = cc[1]
stats = cc[2]
centroids = cc[3]
print("There are %a labels" % numLabels)
output = np.zeros(croppedImg.shape, dtype="uint8")

for i in range(numLabels):
    area = stats[i, cv.CC_STAT_AREA]
    if (area > 1250) and (area < 1000000):
        componentMask = (labels == i).astype("uint8") * 255
        output = cv.bitwise_or(output, componentMask)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)

sure_bg = cv.dilate(closing,kernel,iterations=3)

dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.21*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

cc2 = cv.connectedComponentsWithStats(sure_fg)

numMarkers = cc2[0]
markers = cc2[1]
markerStats = cc2[2]
markerCentroids = cc2[3]



testImage = cv.imread("D:\Bachelor Project\Bachelor\Thresholding\Watershed\watershed{}.tiff".format(0), 0)
ccTest = cv.connectedComponentsWithStats(testImage, 8, cv.CV_32S)
numTestLabels = ccTest[0]
testLabels = ccTest[1]
print(testLabels)
testOutput = np.zeros(testImage.shape, dtype="uint8")
testOutput1 = np.zeros(testImage.shape, dtype="uint8")
componentMask = (testLabels == 0).astype("uint8") * 255
testOutput = cv.bitwise_or(testOutput, componentMask)

markers = markers+1
markers[unknown==255] = 0


result = cv.cvtColor(croppedImg, cv.COLOR_GRAY2BGR)
markers = cv.watershed(result, markers)
result[markers == -1] = [0,255,0]

fig = plt.figure()
rows = 1
cols = 1


fig.add_subplot(rows, cols, 1)
plt.imshow(componentMask, 'gray')
plt.title("img")



plt.show()

#cv.imwrite("D:\Bachelor Project\Bachelor\Thresholding\ConnectedComponentsLipids\connectedComponentsLipids20.tiff", closing)
#cv.imwrite('Thresholding\Watershed\watershed0.tiff', result)
print("done")