import cv2 as cv
from glob import glob
import os
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from matplotlib import pyplot as plt
y_true = []
y_pred = []

ground_truths = glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Chenhaos Unet\Nuclei\Run\Test Truth", "*.tif"))
predictions = glob(os.path.join(r"D:\Bachelor Project\Bachelor\UNET\Segmentations\Nuclei\Test", "*.tif"))

for i in range(len(ground_truths)):
    truth = cv.imread(ground_truths[i], 0)
    pred = cv.imread(predictions[i], 0)

    y_true.append(truth)
    y_pred.append(pred)


flat_true = np.array(y_true).flatten()
flat_pred = np.array(y_pred).flatten()

 

matrix = confusion_matrix(flat_true, flat_pred, normalize="all", labels=[255,0])
score = f1_score(flat_true, flat_pred, pos_label = 255)
print("F1 score:" + str(score))

precision = matrix[1,1] / (matrix[1,1] + matrix[0,1])
recall = matrix[1,1] / (matrix[1,1] + matrix[1,0])

print("Precision:" + str(precision))
print("Recall:" + str(recall))

disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Positive","Negative"])
disp.plot()
plt.show()