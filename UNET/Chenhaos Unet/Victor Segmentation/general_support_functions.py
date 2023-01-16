# -*- coding: utf-8 -*-
"""
General Support Functions - version 1.0

@author: Chenhao Wang
@date: June 2021
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.io as skio

###########################################################################
# folder paths 

root_folder = os.getcwd()

raw_folder = root_folder + '/raw_data'
raw_folder_train_img = raw_folder + "/train" + "/images"
raw_folder_train_mask = raw_folder + "/train" + "/masks"
raw_folder_valid_img = raw_folder + "/valid" + "/images"
raw_folder_valid_mask = raw_folder + "/valid" + "/masks"

raw_data_axon_folder = root_folder + '/raw_data_axon'
raw_data_axon_mask_folder = raw_data_axon_folder + '/masks'
raw_data_axon_img_folder = raw_data_axon_folder + '/images'


cross_val_train_img_folder = root_folder + "\\x_val" + "\\train\\images"
cross_val_train_mask_folder = root_folder + "\\x_val" + "\\train\\masks"
cross_val_valid_img_folder = root_folder + "\\x_val" + "\\valid\\images"
cross_val_valid_mask_folder = root_folder + "\\x_val" + "\\valid\\masks"
cross_val_valid_img_augmented_folder = raw_folder + "\\x_val" + "\\valid_augmented\\images"
cross_val_valid_mask_augmented_folder  = raw_folder + "\\x_val" + "\\valid_augmented\\masks"

cross_val_test_img_folder = root_folder + "\\x_val" + "\\test\\images"
cross_val_test_mask_folder = root_folder + "\\x_val" + "\\test\\prediction"




full_sized_folder = root_folder + "/full_sized_data"
full_sized_folder_train_img = full_sized_folder + "/train" + "/images"
full_sized_folder_train_mask = full_sized_folder + "/train" + "/masks"
full_sized_folder_valid_img = full_sized_folder + "/valid" + "/images"
full_sized_folder_valid_mask = full_sized_folder + "/valid" + "/masks"

cristae_image_folder = root_folder + "/cristae_annotation" + "/images"
cristae_mask_folder = root_folder + "/cristae_annotation" + "/masks"


test_folder = raw_folder + '/test'
results_folder = root_folder + '/results'



###########################################################################
# file management functions

def setpath(path):
    '''
    changes working directory, creates folder if path doesn't exist
    '''
    try:
        os.chdir(path)
    except OSError:
        os.makedirs(path)
        os.chdir(path)           


def all_file_names(folder, file_format = None):
    '''
    list all image filenames of a given format in a folder,
    if file_format == None, returns all files regardless of format.
    '''
    if file_format is None:
        names = os.listdir(folder)
    else:
        names = os.listdir(folder)
        for name in names:
            if name[-len(file_format):] != file_format:
                names.remove(name)
    return names



###########################################################################
# image data loaders



def load_volume_data_slices_different_size(folder_path):
    """
    loads all the image slices in the specified folder of a single image volume. 
    """    
    setpath(folder_path)
    file_names = all_file_names(folder_path)
    output_collection = []
    for name in tqdm(file_names):
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        output_collection.append(img)
    return file_names, output_collection



def load_volume_data_slices_color_different_size(folder_path):
    """
    loads all the image slices in the specified folder of a single image volume. 
    """    
    setpath(folder_path)
    file_names = all_file_names(folder_path)
    output_collection = []
    for name in tqdm(file_names):
        img = cv2.imread(name)
        output_collection.append(img)
    return file_names, output_collection



def load_volume_data_slices(folder_path):
    """
    loads all the image slices in the specified folder of a single image volume. 
    """    
    setpath(folder_path)
    file_names = all_file_names(folder_path)
    output_collection = []
    for name in tqdm(file_names):
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        output_collection.append(img)
    output_collection = np.array(output_collection)
    return file_names, output_collection


def load_volume_data_slices_color(folder_path):
    """
    loads all the image slices in the specified folder of a single image volume. 
    """    
    setpath(folder_path)
    file_names = all_file_names(folder_path)
    output_collection = []
    for name in tqdm(file_names):
        img = cv2.imread(name)
        output_collection.append(img)
    output_collection = np.array(output_collection)
    return file_names, output_collection



def load_volume_data_tif_stacks(input_path,
                                file_name):
    """
    loads all the image slices in the specified folder of a single image volume. 
    """    
    setpath(input_path)
    img_stack = skio.imread(file_name, plugin="tifffile")
    
    
    return img_stack



def save_img_as_slices(volume,
                       output_path,
                       initial_tag = None):
    
    setpath(output_path)

    for i, img_slice in enumerate(tqdm(volume)):
        if i < 10:
            save_name = "0000" + str(i) + ".tif"
        elif i < 100:
            save_name = "000" + str(i) + ".tif"
        elif i < 1000:
            save_name = "00" + str(i) + ".tif"
        elif i < 10000:
            save_name = "0" + str(i) + ".tif"
        else:
            save_name = str(i) + ".tif"
    
        if initial_tag != None:
            save_name = initial_tag + save_name
            
        cv2.imwrite(save_name, img_slice)


def save_img_as_slices_with_names(volume,
                                  output_path,
                                  volume_names):
    
    setpath(output_path)

    for i, img_slice in enumerate(tqdm(volume)):
        save_name = volume_names[i]
        cv2.imwrite(save_name, img_slice)



###########################################################################
# image visualization
    
def visualize_jet(img, max_val = 0.0000005):
    plt.figure(figsize = (15,15))
    #plt.imshow(img, cmap='jet')
    ax = plt.gca()
    im = ax.imshow(img, cmap='jet')
    im.set_clim(vmin = 0, vmax=max_val)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.tick_params(labelsize=12)
    plt.tick_params(labelsize=15)    
    
    
def visualize(img):
    plt.figure(figsize = (15,15))
    plt.imshow(img, cmap='gray')



###########################################################################
# image filtering 

def multiple_slice_gaussian_denoise(image_array,
                                    temporal_size = 3,
                                    filter_strength = 0.5):
    
    """denoise gaussian white noise using Non-local Means Denoising algorithm:
       http://www.ipol.im/pub/art/2011/bcm_nlm/
       
       Multiframe version.
    """
    
    image_array_denoised = []
   
    margin = int(temporal_size/2)
    
    for i in tqdm(range(margin, len(image_array) - margin)):
        # performs multiple frame denoising
        noise_filtered = cv2.fastNlMeansDenoisingMulti(image_array,
                                                       imgToDenoiseIndex = i,
                                                       temporalWindowSize = temporal_size,
                                                       h = filter_strength)
        image_array_denoised.append(noise_filtered)
        
    return np.array(image_array_denoised)




def adaptive_thresholding_gaussian(image_volume,
                                   win_size = 41,
                                   c = 2,
                                   max_intensity_value = 255):

    output = []
    for image in tqdm(image_volume):
        threshold = cv2.adaptiveThreshold(image,
                                          max_intensity_value,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                          cv2.THRESH_BINARY,
                                          win_size,
                                          c)
        threshold = 1 - threshold/threshold.max()
        output.append(threshold)
    return np.array(output)




###########################################################################
# image filtering 

def cut_images_into_patches_2D(input_names, input_volume,
                               output_path,
                               patch_size,
                               patch_step_ratio_v = 2,
                               patch_step_ratio_h = 2):
    
    """
    cuts images into patches:
        patch_step_size = patch_size / patch_step_ratio
        
    """
    
    patch_step_size_v = int((patch_size[0]/patch_step_ratio_v))
    patch_step_size_h = int((patch_size[1]/patch_step_ratio_h))
    


    setpath(output_path)
    for index in tqdm(range(len(input_volume))):
        raw_img = input_volume[index]
        raw_name = input_names[index]
        
        
        img_slice_shape = raw_img.shape
    
        v_starting_coors = np.arange(int((img_slice_shape[0] - patch_size[0])
                                         /patch_step_size_v) + 1)*patch_step_size_v
    
        h_starting_coors = np.arange(int((img_slice_shape[1] - patch_size[1])
                                         /patch_step_size_h) + 1)*patch_step_size_h
        
        
        
        for i in v_starting_coors: 
            for j in h_starting_coors:  
                    img_patch = raw_img[i : i + patch_size[0],
                                        j : j + patch_size[1]]
                 
                    
                    patch_save_name = raw_name.split(".")[0] + "_" + str(i) + "_" + str(j) + "." + raw_name.split(".")[1]
                    cv2.imwrite(patch_save_name, img_patch)
    
    
    
    
def cut_images_into_patches_2D_stack(input_path,
                                     mask_path,
                                     img_file_name,
                                     mask_file_name,
                                     output_path,
                                     patch_size,
                                     patch_step_ratio_v = 2,
                                     patch_step_ratio_h = 2,
                                     intial_tag = "train_img_"):
    
    """
    cuts images into patches:
        patch_step_size = patch_size / patch_step_ratio
        
    """

    input_volume = load_volume_data_tif_stacks(input_path,
                                               img_file_name)
    
    mask_volume = load_volume_data_tif_stacks(mask_path,
                                              mask_file_name)
    
    patch_step_size_v = int((patch_size[0]/patch_step_ratio_v))
    patch_step_size_h = int((patch_size[1]/patch_step_ratio_h))
    
    img_slice_shape = input_volume[0].shape
    
    v_starting_coors = np.arange(int((img_slice_shape[0] - patch_size[0])
                                     /patch_step_size_v) + 1)*patch_step_size_v
    
    h_starting_coors = np.arange(int((img_slice_shape[1] - patch_size[1])
                                     /patch_step_size_h) + 1)*patch_step_size_h
    
    
    setpath(output_path)
    for index in tqdm(range(len(input_volume))):
        raw_img = input_volume[index]
        raw_mask = mask_volume[index]
        
        for i in v_starting_coors: 
            for j in h_starting_coors:  
                    img_patch = raw_img[i : i + patch_size[0],
                                        j : j + patch_size[1]]
                 
                    mask_patch = raw_mask[i : i + patch_size[0],
                                          j : j + patch_size[1]]
                    
                    if np.sum(mask_patch) > 0:
                        patch_save_name = intial_tag + str(index) + "_"  + str(i) + "_" + str(j) + ".png"
                        cv2.imwrite(patch_save_name, img_patch)





def cut_images_into_patches_2D_stack_train_val_split(input_volume,
                                                     output_path,
                                                     patch_size,
                                                     patch_step_ratio_v = 2,
                                                     patch_step_ratio_h = 2,
                                                     intial_tag = "train_img_"):
    

    
    patch_step_size_v = int((patch_size[0]/patch_step_ratio_v))
    patch_step_size_h = int((patch_size[1]/patch_step_ratio_h))
    
    img_slice_shape = input_volume[0].shape
    
    v_starting_coors = np.arange(int((img_slice_shape[0] - patch_size[0])
                                     /patch_step_size_v) + 1)*patch_step_size_v
    
    h_starting_coors = np.arange(int((img_slice_shape[1] - patch_size[1])
                                     /patch_step_size_h) + 1)*patch_step_size_h
    
    
    setpath(output_path)
    for index in tqdm(range(len(input_volume))):
        raw_img = input_volume[index]

        for i in v_starting_coors: 
            for j in h_starting_coors:  
                    img_patch = raw_img[i : i + patch_size[0],
                                        j : j + patch_size[1]]
                 
                    
                    patch_save_name = intial_tag + str(index) + "_"  + str(i) + "_" + str(j) + ".png"
                    cv2.imwrite(patch_save_name, img_patch)




