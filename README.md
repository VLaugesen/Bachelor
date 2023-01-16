Code for bachelor project Segmentation of brain cells from 3d electron microscopy by Victor Bonde Damgaard Laugesen

# File explanation
For the sake of saving space, all images have been removed from the repository.

Thresholding - Everything to do with thresholding the lipids

FraJon - Misc. files sent by Jon Sporring

areas.py - file used for ccmputing and visualising scatterplot on thresholded lipids

lipidsThreshold.py - file performing the thresholding



UNET - Everything to do with UNET:

Chenhaos Unet - Folder containing the UNET implementation by Chenhao Wang

Mitochondria/Run - Contains the final models for segmenting mitochondria

Nuclei/Run - Contains the final models for segmenting nuclei

Victor Segmentation - Folder containing the UNET

img_processing.py - used to augment and distribute images and masks

main.py - file for training and/or segmenting



analysis.py - main file for performing analysis on segmented images

showImage - file used for displaying images



Evaluate.py - Compute confusion matrices, F1 scores, precision, and recall

createImages - Miscelannious functions used to cut, crop, construct or manipulate images

