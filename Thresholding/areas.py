from attr import asdict
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ccs = []
finishedVolumes = []
lipids = 0

for i in range(21):
    img = cv.imread("D:\Bachelor Project\Bachelor\Thresholding\Watershed\watershed{}.tiff".format(i), 0)


    cc = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)

    numLabels = cc[0]
    labels = cc[1]
    stats = cc[2]
    centroids = cc[3]

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

noNext = [-1] * 38
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
                volume += ccs[k][1][curNext] * 8.2 ** 2
                volumeMatrix.append((ccs[k][4] == curNext + 1).astype("uint8") * 255)

                curNext = ccs[k][5][curNext]
                k += 1
            volumeMatrix = np.array(volumeMatrix)
            verts, faces, normals, values = measure.marching_cubes_lewiner(volumeMatrix, 0, spacing=(30, 8.2, 8.2))
            surfaceArea = measure.mesh_surface_area(verts, faces)
            print("lipid number: " + str(lipids))
   


            finishedVolumes.append((volume, surfaceArea))
            lipids += 1

        #elif ccs[i][3][j] == 0:
        #    ccs[i][3][j] = 1
        #    volume = ccs[i][1][j]
        #    finishedVolumes.append(volume)

finishedVolumes = np.array(finishedVolumes)
print(finishedVolumes.shape)


x = finishedVolumes[:,0]
y = finishedVolumes[:,1]

fig = plt.figure()

rows = 1
cols = 1

fig.add_subplot(rows, cols, 1)
plt.scatter(x, y)
plt.xlabel("volume")
plt.ylabel("surfaceArea")
plt.title("scatterplot of surface area vs volume")

#fig.add_subplot(rows, cols, 1)
#plt.hist(areas, bins=50)
#plt.title("histogram of volumes")

plt.show()