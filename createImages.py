import cv2 as cv
import numpy as np
from tifffile import imwrite
from matplotlib import pyplot as plt
from glob import glob
import os

def makeAll():
    result = []

    for i in range(21):
        img = cv.imread(r"D:\Bachelor Project\Bachelor\UNET\Segments\Nuclei\segment" + str(i) + ".tif", 0)
        result.append(img)
    imwrite(r"D:\Bachelor Project\Bachelor\UNET\All.tiff", np.array(result))


def cutImage():
    img = cv.imread(r"D:\Bachelor Project\Bachelor\UNET\Cropped Images\Mitochondria\s2700_e01_cropped0.tif")
    imgMask = cv.imread(r"D:\Bachelor Project\Bachelor\UNET\Cropped Image Masks\Mitocondria\s2700_e01_cropped_mask_mitochondria0.tif")

    for i in range(4):
        for j in range(4):
            x,y,h,w = i * 1250, j * 1250, 1250, 1250

            croppedImg = img[y:y+h, x:x+w]
            croppedImgMask = imgMask[y:y+h, x:x+w]

            cv.imwrite(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Victor Segmentation\victor_data\Images\s2700_e01_cropped0" + str(i) + str(j) + ".tif", croppedImg)
            cv.imwrite(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Victor Segmentation\victor_data\Masks\s2700_e01_cropped_mask_mitochondria0" + str(i) + str(j) + ".tif", croppedImgMask)


def cropImages(x, y, w, h):
    images = sorted(glob(os.path.join(r"D:\Bachelor Project\Bachelor\Data", "*.tif")))
    for i in range(len(images)):
        img = cv.imread(images[i])
        croppedImg = img[y:y+h, x:x+w]
        print(i)
        cv.imwrite(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Victor Segmentation\victor_data\Test_Images\i" + str(i) + ".tif", croppedImg)

def resize(path):
    images = sorted(glob(os.path.join(path, "*.tif")))
    for i in range(len(images)):
        img = cv.imread(images[i])
        newImg = cv.resize(img, dsize=(1184, 1184), interpolation=cv.INTER_AREA)
        cv.imwrite(images[i], newImg)
    
def binurize(path):
    images = glob(os.path.join(path, "*.tif"))
    for i in range(len(images)):
        img = cv.imread(images[i], 0)
        ret, thresh = cv.threshold(img, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        cv.imwrite(r"D:\Bachelor Project\Bachelor\UNET\Segmentations\Nuclei\Test\nuc" + str(i) + ".tif", thresh)

def histogram(path):
    img = cv.imread(path, 0)

    plt.hist(img.ravel(), 255, (1,256))
    plt.title("histogram of fourth slice")

    plt.show()

binurize(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Nuclei\Run\output\masks")

#cropImages(880, 780, 5000, 5000)