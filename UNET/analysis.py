import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import os
from scipy.spatial import distance_matrix
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

masks = glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Mitochondria\Run2\output\masks", "*.tif"))
print(len(masks))
#masks = glob(os.path.join(r"D:\Bachelor Project\Bachelor\Thresholding\Watershed", "*.tiff"))
ccs = []
finishedVolumes = []
nuclei = 0

for i in range(len(masks)):

    img = cv.imread(masks[i], 0)

    ret, thresh = cv.threshold(img, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    cc = cv.connectedComponentsWithStats(closing, 8, cv.CV_32S)
    
    numLabels = cc[0]
    labels = cc[1]
    stats = cc[2]

    output = np.zeros(img.shape, dtype="uint8")
    for j in range(1,numLabels):
        area = stats[j, cv.CC_STAT_AREA]
        if (area > 750):
            componentMask = (labels == j).astype("uint8") * 255
            output = cv.bitwise_or(output, componentMask)
    cv.imwrite(r"D:\Bachelor Project\Bachelor\UNET\Segmentations\Mitochondria\segment" + str(i) + ".tif", output)

    cc2 = cv.connectedComponentsWithStats(img, 8, cv.CV_32S)

    numLabels = cc2[0]
    labels = cc2[1]
    stats = cc2[2]
    centroids = cc2[3]

    numLabels-=1; 
    areas = stats[1:,-1]; 
    centroids=centroids[1:,:]

    visited = [0] * numLabels

    ccs.append([numLabels, areas, centroids, visited, labels])
    
for i in range(20):
    distMat = distance_matrix(ccs[i][2], ccs[i+1][2])
    nexts = []
    for j in range(len(distMat[:])):
        minIndex = np.argmin(distMat[j, :])
        if distMat[j, minIndex] < 40:
            nexts.append(minIndex)
        else:
            nexts.append(-1)
        
    ccs[i].append(nexts)

noNext = [-1] * len(ccs[20][3])
ccs[20].append(noNext)

for i in range(20):
    for j in range(ccs[i][0]):
        if ccs[i][3][j] == 0 and ccs[i][5][j] != -1:
            ccs[i][3][j] = 1
            volume = ccs[i][1][j]
            volumeMatrix = [(ccs[i][4] == j + 1).astype("uint8") * 255]

            curNext = ccs[i][5][j]
            k = i+1
            while(curNext != -1):
                ccs[k][3][curNext] = 1
                volume += ccs[k][1][curNext] * (8.2 ** 2)
                volumeMatrix.append((ccs[k][4] == curNext + 1).astype("uint8") * 255)

                curNext = ccs[k][5][curNext]
                k += 1
            volumeMatrix = np.array(volumeMatrix)
            verts, faces, normals, values = measure.marching_cubes_lewiner(volumeMatrix, 0, spacing=(30, 8.2, 8.2))
            surfaceArea = measure.mesh_surface_area(verts, faces)

            finishedVolumes.append([volume, surfaceArea])
            print("Nucleus number:" + str(nuclei))
            nuclei += 1

finishedVolumes = np.array(finishedVolumes)
print("Number of nuclei:" + str(len(finishedVolumes)))


fig = plt.figure()
rows = 1
cols = 1

fig.add_subplot(rows, cols, 1)
plt.scatter(finishedVolumes[:,0], finishedVolumes[:,1])
plt.xlabel("volume")
plt.ylabel("surfaceArea")
plt.title("scatterplot of surface area vs volume")

#fig.add_subplot(rows, cols, 2)
#plt.hist(areas, bins=5)
#plt.title("histogram of volumes")

plt.show()