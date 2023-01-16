from matplotlib import pyplot as plt
import cv2 as cv
from glob import glob
import os


def showImages():
    imagesMit = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Mitochondria\Run\output\images", "*.tif")))
    imagesNuc = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Nuclei\Run\output\images", "*.tif")))

    truthsMit = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Mitochondria\Run\Test Truth", "*.tif")))
    truthsNuc = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Nuclei\Run\Test truth", "*.tif")))

    predMit = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Mitochondria\Run\output\masks", "*.tif")))
    predNUc = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Nuclei\Run\output\masks", "*.tif")))
    #outputs = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\FinishedRunMany\output\masks", "*.tif")))

    fig = plt.figure()
    rows = 3
    cols = 7
    count = 1

    for i in range(len(imagesMit)):
        img = cv.imread(imagesMit[i])
        fig.add_subplot(rows, cols, count)
        plt.imshow(img, 'gray')
        count += 1

    for i in range(len(imagesNuc)):
        img = cv.imread(imagesNuc[i])
        fig.add_subplot(rows, cols, count)
        plt.imshow(img, 'gray')
        count += 1

    for i in range(len(truthsMit)):
        img = cv.imread(truthsMit[i])
        fig.add_subplot(rows, cols, count)
        plt.imshow(img, 'gray')
        count += 1

    for i in range(len(truthsNuc)):
        img = cv.imread(truthsNuc[i])
        fig.add_subplot(rows, cols, count)
        plt.imshow(img, 'gray')
        count += 1

    for i in range(len(predMit)):
        img = cv.imread(predMit[i])
        fig.add_subplot(rows, cols, count)
        plt.imshow(img, 'gray')
        count += 1

    for i in range(len(predNUc)):
        img = cv.imread(predNUc[i])
        fig.add_subplot(rows, cols, count)
        plt.imshow(img, 'gray')
        count += 1
    plt.show()

showImages()