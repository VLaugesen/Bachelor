# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 23:56:31 2020

@author: CHENHAO WANG
"""

from general_support_functions import *
from deep_neural_networks import *
import random
import imutils
import skimage.morphology

###########################################################################
# image intensity and effects augmentation functions

def noise_mask_normal_randomized(var = 50, 
                                 img_shape = (1,256,256,1),
                                 ratio_of_pixels_to_adjust = 0.15):
    
    deviation = var * random.random()
    
    noise = np.random.normal(0, deviation, img_shape)
    
    ratio_of_pixels_to_use = np.random.choice(np.arange(ratio_of_pixels_to_adjust/2, 
                                                        ratio_of_pixels_to_adjust, 
                                                        0.01))
    
    ratio_mask = np.random.rand(*img_shape) < ratio_of_pixels_to_use
    
    output_noise_mask = noise * ratio_mask
    return output_noise_mask
        
     
def random_intensity_adjustment(input_img):
    """scales image intensity between 0 and 1, and adds random brightness and contrast adjustment"""
    
    img = np.copy(input_img)
    img = img.astype(np.float)
    

    # first channel
    alpha = np.random.choice(np.arange(0.8, 1.21, 0.05))
    beta = np.random.choice(np.arange(-15, 16, 2))
    img = img * alpha + beta
    
    
    # adds or reduces noise
    noise_mask = noise_mask_normal_randomized(var = 35, 
                                              img_shape = img.shape,
                                              ratio_of_pixels_to_adjust = 0.075)

    # applies the noise
    img = img + noise_mask


    # regulates the intensity range
    img = img/255
    img[img > 1] = 1
    img[img < 0] = 0



    # completely black images will stay unaugmented
    if input_img.sum() == 0:
        img = np.copy(input_img)/255
   
    
    return img



def random_intensity_adjustment_no_noise(input_img):
    """scales image intensity between 0 and 1, and adds random brightness and contrast adjustment"""
    
    img = np.copy(input_img)
    img = img.astype(np.float)
    

    # first channel
    alpha = np.random.choice(np.arange(0.8, 1.21, 0.05))
    beta = np.random.choice(np.arange(-15, 16, 2))
    img = img * alpha + beta
    

    # regulates the intensity range
    img = img/255
    img[img > 1] = 1
    img[img < 0] = 0


    # completely black images will stay unaugmented
    if input_img.sum() == 0:
        img = np.copy(input_img)/255
   
    
    return img


###########################################################################
# image and mask generator


def batch_reshaper(file_names, 
                   batch_size, 
                   epoch):
    names = np.array(file_names)
    total_names_count = int(len(file_names)/batch_size)*batch_size
    names = names[:total_names_count]
    names = names.reshape((int(total_names_count/batch_size), batch_size))
    names = names.tolist() * epoch
    names = np.array(names)
    return names
        


def prepare_training_generator(batch_size,
                               epochs,
                               input_img_path,
                               input_mask_path, 
                               patch_size):
    
    train_img_names = np.array(all_file_names(input_img_path))
    train_mask_names = np.array(all_file_names(input_mask_path))
    
    randomized_order = np.arange(len(train_img_names))
    np.random.shuffle(randomized_order)
    shuffled_train_img_names = train_img_names[randomized_order]
    shuffled_train_mask_names = train_mask_names[randomized_order]
    
    shuffled_train_img_names = batch_reshaper(shuffled_train_img_names, 
                                              batch_size, 
                                              epochs)
    
    shuffled_train_mask_names = batch_reshaper(shuffled_train_mask_names, 
                                               batch_size, 
                                               epochs)
    
    

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 33, 
                                                              width_shift_range=0.2, 
                                                              height_shift_range=0.2, 
                                                              shear_range=0.2, 
                                                              zoom_range=0.2,  
                                                              fill_mode='constant',
                                                              horizontal_flip=True, 
                                                              vertical_flip=True) 
    
    mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 33, 
                                                               width_shift_range=0.2, 
                                                               height_shift_range=0.2, 
                                                               shear_range=0.2, 
                                                               zoom_range=0.2, 
                                                               fill_mode='constant',
                                                               horizontal_flip=True, 
                                                               vertical_flip=True)
    
    
    # generator steps
    for i in range(len(shuffled_train_img_names)):
        
        batch_of_inputs = []
        batch_of_masks = []

        batch_names_img = shuffled_train_img_names[i]
        batch_names_mask = shuffled_train_mask_names[i]
        
        
        for batch_instance_index in range(batch_size):
            # loads the data
            current_load_name_img = batch_names_img[batch_instance_index]
            current_img_instance = cv2.imread(input_img_path + "/" + current_load_name_img, 
                                              cv2.IMREAD_GRAYSCALE)
            current_img_instance = cv2.resize(current_img_instance, 
                                              dsize=(patch_size[0], patch_size[1]), 
                                              interpolation=cv2.INTER_AREA) 
            
            
            current_load_name_mask = batch_names_mask[batch_instance_index]
            current_mask_instance = cv2.imread(input_mask_path + "/" + current_load_name_mask, 
                                               cv2.IMREAD_GRAYSCALE)
            current_mask_instance = current_mask_instance > 100
            current_mask_instance = cv2.resize(current_mask_instance.astype(np.uint8), 
                                               dsize=(patch_size[0], patch_size[1]), 
                                               interpolation=cv2.INTER_AREA)             
            
        
            # puts them into batches
            batch_of_inputs.append(current_img_instance)
            batch_of_masks.append(current_mask_instance)

        batch_of_inputs = np.array(batch_of_inputs)[:,:,:,np.newaxis]
        batch_of_masks = np.array(batch_of_masks)[:,:,:,np.newaxis]
        
        
        seed = np.random.randint(100000)
        img_generator = img_gen.flow(batch_of_inputs, 
                                     batch_size = batch_size,
                                     shuffle = False,
                                     seed=seed)
    
        mask_generator = mask_gen.flow(batch_of_masks, 
                                       batch_size = batch_size, 
                                       shuffle = False,
                                       seed=seed)
            
        batch_of_inputs = next(img_generator)
        batch_of_masks = next(mask_generator)
        
        batch_of_inputs = batch_of_inputs.astype(np.float32)
        batch_of_masks = batch_of_masks.astype(np.float32)
        
        for img_index in range(batch_size):
            batch_of_inputs[img_index,:,:,0] = random_intensity_adjustment(batch_of_inputs[img_index,:,:,0])
        
        
        yield batch_of_inputs, batch_of_masks
    
    
    

def prepare_validation_generator(batch_size,
                                 epochs,
                                 input_img_path,
                                 input_mask_path,
                                 patch_size):
    
    train_img_names = np.array(all_file_names(input_img_path))
    train_mask_names = np.array(all_file_names(input_mask_path))
    
    randomized_order = np.arange(len(train_img_names))
    np.random.shuffle(randomized_order)
    shuffled_train_img_names = train_img_names[randomized_order]
    shuffled_train_mask_names = train_mask_names[randomized_order]
    
    shuffled_train_img_names = batch_reshaper(shuffled_train_img_names, 
                                              batch_size, 
                                              epochs)
    
    shuffled_train_mask_names = batch_reshaper(shuffled_train_mask_names, 
                                               batch_size, 
                                               epochs)
    
    
    
    # generator steps
    for i in range(len(shuffled_train_img_names)):
        
        batch_of_inputs = []
        batch_of_masks = []

        batch_names_img = shuffled_train_img_names[i]
        batch_names_mask = shuffled_train_mask_names[i]
        
        
        for batch_instance_index in range(batch_size):
            # loads the data
            current_load_name_img = batch_names_img[batch_instance_index]
            current_img_instance = cv2.imread(input_img_path + "/" + current_load_name_img, 
                                              cv2.IMREAD_GRAYSCALE)
            current_img_instance = cv2.resize(current_img_instance, 
                                              dsize=(patch_size[0], patch_size[1]), 
                                              interpolation=cv2.INTER_AREA) 
            
            
            current_load_name_mask = batch_names_mask[batch_instance_index]
            current_mask_instance = cv2.imread(input_mask_path + "/" + current_load_name_mask, 
                                               cv2.IMREAD_GRAYSCALE)
            current_mask_instance = current_mask_instance > 100
            current_mask_instance = cv2.resize(current_mask_instance.astype(np.uint8), 
                                               dsize=(patch_size[0], patch_size[1]), 
                                               interpolation=cv2.INTER_AREA)  
            
        
            # puts them into batches
            batch_of_inputs.append(current_img_instance)
            batch_of_masks.append(current_mask_instance)

        batch_of_inputs = np.array(batch_of_inputs)[:,:,:,np.newaxis]
        batch_of_masks = np.array(batch_of_masks)[:,:,:,np.newaxis]
        
        
        batch_of_inputs = batch_of_inputs.astype(np.float32)/255
        batch_of_masks = batch_of_masks.astype(np.float32)
        
        
        yield batch_of_inputs, batch_of_masks


###########################################################################
def prepare_multiclass_training_generator(batch_size,
                                          epochs,
                                          input_img_path,
                                          input_mask_path):
    
    train_img_names = np.array(all_file_names(input_img_path))
    train_mask_names = np.array(all_file_names(input_mask_path))
    
    randomized_order = np.arange(len(train_img_names))
    np.random.shuffle(randomized_order)
    shuffled_train_img_names = train_img_names[randomized_order]
    shuffled_train_mask_names = train_mask_names[randomized_order]
    
    shuffled_train_img_names = batch_reshaper(shuffled_train_img_names, 
                                              batch_size, 
                                              epochs)
    
    shuffled_train_mask_names = batch_reshaper(shuffled_train_mask_names, 
                                               batch_size, 
                                               epochs)
    
    

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 20, 
                                                              width_shift_range=0.15, 
                                                              height_shift_range=0.15, 
                                                              shear_range=0.15, 
                                                              zoom_range=0.15,  
                                                              fill_mode='constant',
                                                              horizontal_flip=True, 
                                                              vertical_flip=True) 
    
    
    
    mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 20, 
                                                               width_shift_range=0.15, 
                                                               height_shift_range=0.15, 
                                                               shear_range=0.15, 
                                                               zoom_range=0.15, 
                                                               fill_mode='constant',
                                                               horizontal_flip=True, 
                                                               vertical_flip=True)
    
    
    # generator steps
    for i in range(len(shuffled_train_img_names)):
        
        batch_of_inputs = []
        batch_of_masks = []

        batch_names_img = shuffled_train_img_names[i]
        batch_names_mask = shuffled_train_mask_names[i]
        
        
        for batch_instance_index in range(batch_size):
            # loads the data
            current_load_name_img = batch_names_img[batch_instance_index]
            current_img_instance = cv2.imread(input_img_path + "/" + current_load_name_img, 
                                              cv2.IMREAD_GRAYSCALE)
            current_img_instance = cv2.resize(current_img_instance, 
                                              dsize=(256, 256), 
                                              interpolation=cv2.INTER_AREA) 
            
            
            current_load_name_mask = batch_names_mask[batch_instance_index]
            current_mask_instance = cv2.imread(input_mask_path + "/" + current_load_name_mask, 
                                               cv2.IMREAD_GRAYSCALE)
            current_mask_instance = current_mask_instance > 100
            current_mask_instance = cv2.resize(current_mask_instance.astype(np.uint8), 
                                               dsize=(256, 256), 
                                               interpolation=cv2.INTER_AREA)             
            
            
            current_mask_instance_filled = skimage.morphology.remove_small_holes(current_mask_instance, 
                                                                                 area_threshold=10000, 
                                                                                 connectivity=1)
            
            current_lumen_mask = current_mask_instance_filled - current_mask_instance 
            current_membrane_mask = current_mask_instance
            current_background_mask = 1 - current_mask_instance_filled
        
            
            current_mask_one_hot = np.zeros((current_lumen_mask.shape[0],
                                             current_lumen_mask.shape[1],
                                             3))
            current_mask_one_hot[:,:,0] = current_background_mask
            current_mask_one_hot[:,:,1] = current_membrane_mask
            current_mask_one_hot[:,:,2] = current_lumen_mask
            
            
            # puts them into batches
            batch_of_inputs.append(current_img_instance)
            batch_of_masks.append(current_mask_one_hot)

        batch_of_inputs = np.array(batch_of_inputs)[:,:,:,np.newaxis]
        batch_of_masks = np.array(batch_of_masks)
        
        
        seed = np.random.randint(100000)
        img_generator = img_gen.flow(batch_of_inputs, 
                                     batch_size = batch_size,
                                     shuffle = False,
                                     seed=seed)
    
        mask_generator = mask_gen.flow(batch_of_masks, 
                                       batch_size = batch_size, 
                                       shuffle = False,
                                       seed=seed)
            
        batch_of_inputs = next(img_generator)
        batch_of_masks = next(mask_generator)
        
        batch_of_inputs = batch_of_inputs.astype(np.float32)
        batch_of_masks = batch_of_masks.astype(np.float32)
        
        batch_of_masks[np.where(np.sum(batch_of_masks, axis = -1) == 0)] = [1,0,0]
        
        
        batch_of_masks[:,:,:,1] = batch_of_masks[:,:,:,1] > 0.33
        
        batch_of_masks[:,:,:,2] = batch_of_masks[:,:,:,2] > 0
        
        batch_of_masks[:,:,:,2] = batch_of_masks[:,:,:,2] * (1 - batch_of_masks[:,:,:,1])
        
        
        for img_index in range(batch_size):
            batch_of_inputs[img_index,:,:,0] = random_intensity_adjustment(batch_of_inputs[img_index,:,:,0])
        
        
        yield batch_of_inputs, batch_of_masks
    
    


def prepare_multiclass_validation_generator(batch_size,
                                            epochs,
                                            input_img_path,
                                            input_mask_path):
    
    train_img_names = np.array(all_file_names(input_img_path))
    train_mask_names = np.array(all_file_names(input_mask_path))
    
    randomized_order = np.arange(len(train_img_names))
    np.random.shuffle(randomized_order)
    shuffled_train_img_names = train_img_names[randomized_order]
    shuffled_train_mask_names = train_mask_names[randomized_order]
    
    shuffled_train_img_names = batch_reshaper(shuffled_train_img_names, 
                                              batch_size, 
                                              epochs)
    
    shuffled_train_mask_names = batch_reshaper(shuffled_train_mask_names, 
                                               batch_size, 
                                               epochs)
    
    
    # generator steps
    for i in range(len(shuffled_train_img_names)):
        
        batch_of_inputs = []
        batch_of_masks = []

        batch_names_img = shuffled_train_img_names[i]
        batch_names_mask = shuffled_train_mask_names[i]
        
        
        for batch_instance_index in range(batch_size):
            # loads the data
            current_load_name_img = batch_names_img[batch_instance_index]
            current_img_instance = cv2.imread(input_img_path + "/" + current_load_name_img, 
                                              cv2.IMREAD_GRAYSCALE)
            current_img_instance = cv2.resize(current_img_instance, 
                                              dsize=(256, 256), 
                                              interpolation=cv2.INTER_AREA) 
            
            
            current_load_name_mask = batch_names_mask[batch_instance_index]
            current_mask_instance = cv2.imread(input_mask_path + "/" + current_load_name_mask, 
                                               cv2.IMREAD_GRAYSCALE)
            current_mask_instance = current_mask_instance > 100
            current_mask_instance = cv2.resize(current_mask_instance.astype(np.uint8), 
                                               dsize=(256, 256), 
                                               interpolation=cv2.INTER_AREA)             
            
            
            current_mask_instance_filled = skimage.morphology.remove_small_holes(current_mask_instance, 
                                                                                 area_threshold=10000, 
                                                                                 connectivity=1)
            
            current_lumen_mask = current_mask_instance_filled - current_mask_instance 
            current_membrane_mask = current_mask_instance
            current_background_mask = 1 - current_mask_instance_filled
        
            
            current_mask_one_hot = np.zeros((current_lumen_mask.shape[0],
                                             current_lumen_mask.shape[1],
                                             3))
            current_mask_one_hot[:,:,0] = current_background_mask
            current_mask_one_hot[:,:,1] = current_membrane_mask
            current_mask_one_hot[:,:,2] = current_lumen_mask
            
            
            # puts them into batches
            batch_of_inputs.append(current_img_instance)
            batch_of_masks.append(current_mask_one_hot)


        batch_of_inputs = np.array(batch_of_inputs)[:,:,:,np.newaxis]
        batch_of_masks = np.array(batch_of_masks)
        
        
        batch_of_inputs = batch_of_inputs.astype(np.float32)/255
        batch_of_masks = batch_of_masks.astype(np.float32)
        
        
        yield batch_of_inputs, batch_of_masks



def perform_val_aug(inputs,
                    masks):

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 20, 
                                                              width_shift_range=0.15, 
                                                              height_shift_range=0.15, 
                                                              shear_range=0.15, 
                                                              zoom_range=0.15,  
                                                              fill_mode='constant',
                                                              horizontal_flip=True, 
                                                              vertical_flip=True) 
    
    mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range = 20, 
                                                               width_shift_range=0.15, 
                                                               height_shift_range=0.15, 
                                                               shear_range=0.15, 
                                                               zoom_range=0.15, 
                                                               fill_mode='constant',
                                                               horizontal_flip=True, 
                                                               vertical_flip=True)
    
    seed = np.random.randint(100000)
    img_generator = img_gen.flow(inputs, 
                                 batch_size = len(inputs),
                                 shuffle = False,
                                 seed=seed)
    
    mask_generator = mask_gen.flow(masks, 
                                   batch_size = len(masks), 
                                   shuffle = False,
                                   seed=seed)
            
    batch_of_inputs = next(img_generator)
    batch_of_masks = next(mask_generator)


    batch_of_inputs = batch_of_inputs.astype(np.float32)
    batch_of_masks = batch_of_masks.astype(np.float32)
        
    batch_of_masks[np.where(np.sum(batch_of_masks, axis = -1) == 0)] = [1,0,0]
        
        
    batch_of_masks[:,:,:,1] = batch_of_masks[:,:,:,1] > 0.33
        
    batch_of_masks[:,:,:,2] = batch_of_masks[:,:,:,2] > 0
        
    batch_of_masks[:,:,:,2] = batch_of_masks[:,:,:,2] * (1 - batch_of_masks[:,:,:,1])


        
    for img_index in range(len(inputs)):
        batch_of_inputs[img_index,:,:,0] = random_intensity_adjustment_no_noise(batch_of_inputs[img_index,:,:,0])
        
    return batch_of_inputs, batch_of_masks




def validation_augmentator(input_img_path,
                           input_mask_path,
                           output_path_img,
                           output_path_mask,
                           scale = 5):
    
    train_img_names = np.array(all_file_names(input_img_path))
    train_mask_names = np.array(all_file_names(input_mask_path))
    
    
    # generator steps
    inputs = []
    masks = []
    
    inputs_collection = []
    masks_collection = []
    
    for i in range(len(train_img_names)):
        img_name = train_img_names[i]
        mask_name = train_mask_names[i]
        

        current_img_instance = cv2.imread(input_img_path + "/" + img_name, 
                                          cv2.IMREAD_GRAYSCALE)
        current_img_instance = cv2.resize(current_img_instance, 
                                          dsize=(256, 256), 
                                          interpolation=cv2.INTER_AREA) 
            
         
        current_mask_instance = cv2.imread(input_mask_path + "/" + mask_name, 
                                           cv2.IMREAD_GRAYSCALE)
        current_mask_instance = current_mask_instance > 100
        current_mask_instance = cv2.resize(current_mask_instance.astype(np.uint8), 
                                           dsize=(256, 256), 
                                           interpolation=cv2.INTER_AREA)             
            
        
        current_mask_instance_filled = skimage.morphology.remove_small_holes(current_mask_instance, 
                                                                             area_threshold=10000, 
                                                                             connectivity=1)
            
        current_lumen_mask = current_mask_instance_filled - current_mask_instance 
        current_membrane_mask = current_mask_instance
        current_background_mask = 1 - current_mask_instance_filled
        
            
        current_mask_one_hot = np.zeros((current_lumen_mask.shape[0],
                                         current_lumen_mask.shape[1],
                                         3))
        current_mask_one_hot[:,:,0] = current_background_mask
        current_mask_one_hot[:,:,1] = current_membrane_mask
        current_mask_one_hot[:,:,2] = current_lumen_mask
        
        
        inputs.append(current_img_instance)
        masks.append(current_mask_one_hot)
        
        
    inputs = np.array(inputs)[:,:,:,np.newaxis]
    masks = np.array(masks)  
        
    for img_index in range(len(inputs)):
        inputs_collection.append(inputs[img_index]/255)
        masks_collection.append(masks[img_index])
        
    for iteration in tqdm(range(scale - 1)):
        augmented_inputs, augmented_masks = perform_val_aug(inputs,
                                                            masks)
        for img_index in range(len(augmented_inputs)):
            inputs_collection.append(augmented_inputs[img_index])
            masks_collection.append(augmented_masks[img_index])

    inputs_collection = np.array(inputs_collection)[:,:,:,0]
    masks_collection = np.array(masks_collection)


    save_img_as_slices(inputs_collection*255,
                       output_path_img,
                       initial_tag = None)


    save_img_as_slices(masks_collection*255,
                       output_path_mask,
                       initial_tag = None)


def prepare_multiclass_validation_generator_augmented(batch_size,
                                                      epochs,
                                                      input_img_path,
                                                      input_mask_path):
    
    train_img_names = np.array(all_file_names(input_img_path))
    train_mask_names = np.array(all_file_names(input_mask_path))
    
    randomized_order = np.arange(len(train_img_names))
    np.random.shuffle(randomized_order)
    shuffled_train_img_names = train_img_names[randomized_order]
    shuffled_train_mask_names = train_mask_names[randomized_order]
    
    shuffled_train_img_names = batch_reshaper(shuffled_train_img_names, 
                                              batch_size, 
                                              epochs)
    
    shuffled_train_mask_names = batch_reshaper(shuffled_train_mask_names, 
                                               batch_size, 
                                               epochs)
    
    
    # generator steps
    for i in range(len(shuffled_train_img_names)):
        
        batch_of_inputs = []
        batch_of_masks = []

        batch_names_img = shuffled_train_img_names[i]
        batch_names_mask = shuffled_train_mask_names[i]
        
        
        for batch_instance_index in range(batch_size):
            # loads the data
            current_load_name_img = batch_names_img[batch_instance_index]
            current_img_instance = cv2.imread(input_img_path + "/" + current_load_name_img, 
                                              cv2.IMREAD_GRAYSCALE)

            current_load_name_mask = batch_names_mask[batch_instance_index]
            current_mask_instance = cv2.imread(input_mask_path + "/" + current_load_name_mask)
                    
            
            # puts them into batches
            batch_of_inputs.append(current_img_instance)
            batch_of_masks.append(current_mask_instance)


        batch_of_inputs = np.array(batch_of_inputs)[:,:,:,np.newaxis]
        batch_of_masks = np.array(batch_of_masks)
        
        
        batch_of_inputs = batch_of_inputs.astype(np.float32)/255
        batch_of_masks = batch_of_masks.astype(np.float32)/255
        
        
        yield batch_of_inputs, batch_of_masks


###########################################################################
# multiplanar segmentation


def perform_segmentation_cube_2D(input_array,
                                 UNet_model):
    return UNet_model.predict(input_array[:,:,:,np.newaxis])


def perform_segmentation_cube_3D(input_array,
                                 UNet_model):
    return UNet_model.predict(input_array[np.newaxis,:,:,:,np.newaxis])[0]



def perform_segmentation(input_array,
                         UNet_model,
                         overlap_factor = 3,
                         patch_size = (256, 256, 256, 3),
                         segmentation_mode = "2D",
                         visualization_folder = None):
    
    """performs the segmentation"""

    # performs overlapping segmentation
    result = []
    
    Z = input_array.shape[0]
    X = input_array.shape[1]
    Y = input_array.shape[2]
    
    patch_z = patch_size[0]
    patch_x = patch_size[1]
    patch_y = patch_size[2]
    
    z_factor = int(Z/patch_z)
    x_factor = int(X/patch_x)
    y_factor = int(Y/patch_y)
        
    
    division_tracker = np.zeros((Z, X, Y, patch_size[3]), np.float16)
    prediction_tracker = np.zeros((Z, X, Y, patch_size[3]), np.float16)
    
    i_indices = list(range(int((Z-patch_z)/(patch_z/overlap_factor) + 1)))
    j_indices = list(range(int((X-patch_x)/(patch_x/overlap_factor) + 1)))
    k_indices = list(range(int((Y-patch_y)/(patch_y/overlap_factor) + 1)))
    
    z_criteria = i_indices[-1]*patch_z/overlap_factor + patch_z
    x_criteria = j_indices[-1]*patch_x/overlap_factor + patch_x
    y_criteria = k_indices[-1]*patch_y/overlap_factor + patch_y
    
    
    if z_criteria < Z:
        i_indices.append(-1)
    if x_criteria < X:
        j_indices.append(-1)
    if y_criteria < Y:
        k_indices.append(-1)    
        
        
    for i in i_indices: 
        for j in j_indices:  
            for k in k_indices:  
                if i == -1:
                    start_i = Z - patch_z
                    end_i = Z
                else:
                    start_i = int(i*patch_z/overlap_factor)
                    end_i = int(i*patch_z/overlap_factor + patch_z)
                    
                if j == -1:
                    start_j = X - patch_x
                    end_j = X
                else:
                    start_j = int(j*patch_x/overlap_factor)
                    end_j = int(j*patch_x/overlap_factor + patch_x)                    
                    
                if k == -1:
                    start_k = Y - patch_y
                    end_k = Y
                else:
                    start_k = int(k*patch_y/overlap_factor)
                    end_k = int(k*patch_y/overlap_factor + patch_y)                       
                    
                    
                img_patch = input_array[start_i : end_i,
                                        start_j : end_j,
                                        start_k : end_k]

                img_patch = img_patch/255
                
                if segmentation_mode == "2D":
                    segmented_patch = perform_segmentation_cube_2D(img_patch,
                                                                   UNet_model)
                else:
                    segmented_patch = perform_segmentation_cube_3D(img_patch,
                                                                   UNet_model)
                    
                # drops the unreliable near boundary predictions
                if i == 0 or i == int((Z-patch_z)/(patch_z/overlap_factor)) or i == -1 or j == 0 or j == int((X-patch_x)/(patch_x/overlap_factor)) or j == -1 or k == 0 or k == int((Y-patch_y)/(patch_y/overlap_factor)) or k == -1 or overlap_factor < 3:
                    drop_half_factor = 0
                else:
                    drop_half_factor = 0.15

                boundary_mask = np.zeros(segmented_patch.shape)
                z_drop_half = int(segmented_patch.shape[0] * drop_half_factor)
                x_drop_half = int(segmented_patch.shape[1] * drop_half_factor)
                y_drop_half = int(segmented_patch.shape[2] * drop_half_factor)
                boundary_mask[z_drop_half : segmented_patch.shape[0] - z_drop_half,
                              x_drop_half : segmented_patch.shape[1] - x_drop_half,
                              y_drop_half : segmented_patch.shape[2] - y_drop_half] = 1
                segmented_patch = segmented_patch * boundary_mask
                
                division_tracker_patch = boundary_mask
                
                division_tracker[start_i : end_i,
                                 start_j : end_j,
                                 start_k : end_k] = division_tracker[start_i : end_i,
                                                                     start_j : end_j,
                                                                     start_k : end_k] + division_tracker_patch
                prediction_tracker[start_i : end_i,
                                   start_j : end_j,
                                   start_k : end_k] = prediction_tracker[start_i : end_i,
                                                                         start_j : end_j,
                                                                         start_k : end_k] + segmented_patch 
        
        prediction = prediction_tracker / division_tracker
    
    
    prediction = prediction.astype(np.float64)
    
    for l in range(len(prediction)):
        # visualization
        if l < 10:
            str_name = "0000" + str(l)
        elif l < 100:
            str_name = "000" + str(l)
        elif l < 1000:
            str_name = "00" + str(l)
        elif l < 10000:
            str_name = "0" + str(l) 
        else:
            str_name = str(l)
        save_name = str_name + "_seg.png"
        save_name_img = str_name + "_img.png"
        
        if visualization_folder != None:
            setpath(visualization_folder + "/segmentations")
            cv2.imwrite(save_name, (prediction[l]*255).astype(np.uint8))
            setpath(visualization_folder + "/images")
            cv2.imwrite(save_name_img, input_array[l])
            
    return prediction



def rotate(input_data,
           angle): 
    """rotates a 3D array of images in along the first 2 axes."""  
    original = np.ones(input_data[0].shape)
    output = np.array([imutils.rotate_bound(img, angle) for img in input_data])
    output = output.astype(np.uint8)
    return output, original.shape



def rotate_reverse(target_data,
                   original_shape,
                   angle): 
    """rotates a 3D array of images in along the first 2 axes."""  
    half_v, half_h = int(original_shape[0]/2), int(original_shape[1]/2)
    rotated_mask_0 = imutils.rotate_bound(target_data[0], -angle)
    shape = rotated_mask_0.shape
    centre_v, centre_h = int(shape[0]/2), int(shape[1]/2)
    output = []
    for mask in target_data:
        rotated_mask = imutils.rotate_bound(mask, -angle)[centre_v - half_v : centre_v + half_v,
                                                          centre_h - half_h : centre_h + half_h]
        output.append(rotated_mask)                                              
    output = np.array(output)
    return output



def orientation_changer(input_data,
                        axis = "horizontal",
                        reverse = False):
    
    """
    axes = [0,1] : flip vertically
    axes = [1,2] : rotate without flipping
    axes = [0,2] : flip horizontally
    """
    
    if reverse == False:
        rotate_count = 1
    else:
        rotate_count = 3
    
    if axis == "horizontal":
        output = np.rot90(input_data, rotate_count, axes= [0,2])
    else:
        output = np.rot90(input_data, rotate_count, axes= [0,1])
    
    
    return output



def rotated_orientation_changer(input_data):
    
    rotated_data, original_shape = rotate(input_data, angle = 45)
    
    data_A = orientation_changer(rotated_data,
                                 axis = "horizontal",
                                 reverse = False)
    
    data_B = orientation_changer(rotated_data,
                                 axis = "vertical",
                                 reverse = False)
    
    del rotated_data
    
    return data_A, data_B, original_shape
    
    
def rotated_orientation_reversal_A(A, shape):
    
    rotated_data_A = orientation_changer(A,
                                         axis = "horizontal",
                                         reverse = True)

    output = rotate_reverse(rotated_data_A,
                            shape,
                            angle = 45)
    
    return output



def rotated_orientation_reversal_B(B, shape):
    
    rotated_data_B = orientation_changer(B,
                                         axis = "vertical",
                                         reverse = True)

    output = rotate_reverse(rotated_data_B,
                            shape,
                            angle = 45)
    
    return output



def perform_multiplanar_cube_segmentation(input_array,
                                          UNet_model,
                                          patch_size = (256, 256, 256, 3)):


    ######################################## original orientation ########################################
    input_array_O = np.copy(input_array)
    result_O = perform_segmentation(input_array_O,
                                    UNet_model,
                                    overlap_factor = 3,
                                    patch_size = patch_size,
                                    segmentation_mode = "2D",
                                    visualization_folder = None)
    o_A, o_B, o_shape = rotated_orientation_changer(input_array_O)
    

    
    result_O_A = perform_segmentation(o_A,
                                      UNet_model,
                                      overlap_factor = 3,
                                      patch_size = patch_size,
                                      segmentation_mode = "2D",
                                      visualization_folder = None)
    result_O_A = rotated_orientation_reversal_A(result_O_A, o_shape)
    
    if len(result_O_A.shape)<4:
        result_O_A = result_O_A[:,:,:,np.newaxis]
        
        

    result_O_B = perform_segmentation(o_B,
                                      UNet_model,
                                      overlap_factor = 3,
                                      patch_size = patch_size,
                                      segmentation_mode = "2D",
                                      visualization_folder = None)

    result_O_B = rotated_orientation_reversal_B(result_O_B, o_shape)

    if len(result_O_B.shape)<4:
        result_O_B = result_O_B[:,:,:,np.newaxis]

    ######################################## vertical orientation ########################################

    input_array_V = orientation_changer(input_array,
                                        axis = "vertical",
                                        reverse = False) 
    result_V = perform_segmentation(input_array_V,
                                    UNet_model,
                                    overlap_factor = 3,
                                    patch_size = patch_size,
                                    segmentation_mode = "2D",
                                    visualization_folder = None)
    result_V = orientation_changer(result_V,
                                   axis = "vertical",
                                   reverse = True) 
    

    v_A, v_B, v_shape = rotated_orientation_changer(input_array_V)
        
    result_V_A = perform_segmentation(v_A,
                                      UNet_model,
                                      overlap_factor = 3,
                                      patch_size = patch_size,
                                      segmentation_mode = "2D",
                                      visualization_folder = None)
    result_V_A = rotated_orientation_reversal_A(result_V_A, v_shape)
    result_V_A = orientation_changer(result_V_A,
                                     axis = "vertical",
                                     reverse = True) 
    if len(result_V_A.shape)<4:
        result_V_A = result_V_A[:,:,:,np.newaxis]
        
        
    result_V_B = perform_segmentation(v_B,
                                      UNet_model,
                                      overlap_factor = 3,
                                      patch_size = patch_size,
                                      segmentation_mode = "2D",
                                      visualization_folder = None)
    result_V_B = rotated_orientation_reversal_B(result_V_B, v_shape)
    result_V_B = orientation_changer(result_V_B,
                                     axis = "vertical",
                                     reverse = True) 
    if len(result_V_B.shape)<4:
        result_V_B = result_V_B[:,:,:,np.newaxis]



    ######################################## horizontal orientation ########################################

    input_array_H = orientation_changer(input_array,
                                            axis = "horizontal",
                                            reverse = False) 
    result_H = perform_segmentation(input_array_H,
                                    UNet_model,
                                    overlap_factor = 3,
                                    patch_size = patch_size,
                                    segmentation_mode = "2D",
                                    visualization_folder = None)
    result_H = orientation_changer(result_H,
                                   axis = "horizontal",
                                   reverse = True) 



    h_A, h_B, h_shape = rotated_orientation_changer(input_array_H)
    
    result_H_A = perform_segmentation(h_A,
                                      UNet_model,
                                      overlap_factor = 3,
                                      patch_size = patch_size,
                                      segmentation_mode = "2D",
                                      visualization_folder = None)
    result_H_A = rotated_orientation_reversal_A(result_H_A, h_shape)
    result_H_A = orientation_changer(result_H_A,
                                     axis = "horizontal",
                                     reverse = True) 
    if len(result_H_A.shape)<4:
        result_H_A = result_H_A[:,:,:,np.newaxis]
        
        
    result_H_B = perform_segmentation(h_B,
                                      UNet_model,
                                      overlap_factor = 3,
                                      patch_size = patch_size,
                                      segmentation_mode = "2D",
                                      visualization_folder = None)
    result_H_B = rotated_orientation_reversal_B(result_H_B, h_shape)
    result_H_B = orientation_changer(result_H_B,
                                     axis = "horizontal",
                                     reverse = True) 
    if len(result_H_B.shape)<4:
        result_H_B = result_H_B[:,:,:,np.newaxis]
    

    ######################################## merges results ########################################

    multiplanar_result = (result_O + result_O_A + result_O_B
                         +result_V + result_V_A + result_V_B
                         +result_H + result_H_A + result_H_B)/9
    
    
    return multiplanar_result




def perform_memory_efficient_multiplanar_segmentation_large(input_array,
                                                            model_save_name,
                                                            sub_vol_start_index = 0,
                                                            sub_vol_end_index = None,
                                                            custom_loss = 'multiclass_IOU_loss',
                                                            overlap_factor = 3,
                                                            patch_size = (256, 256, 256, 3),
                                                            results_folder = '/results'):
    
    """performs the segmentation"""

    setpath(root_folder)
    if custom_loss == 'multiclass_IOU_loss':
        UNet_model = tf.keras.models.load_model(model_save_name, 
                                                custom_objects={'multiclass_IOU_loss': multiclass_IOU_loss})
    else:
        UNet_model = tf.keras.models.load_model(model_save_name, 
                                                custom_objects={'IOU_loss': IOU_loss})

    # performs overlapping segmentation
    result = []
    
    Z = input_array.shape[0]
    X = input_array.shape[1]
    Y = input_array.shape[2]
    
    patch_z = patch_size[0]
    patch_x = patch_size[1]
    patch_y = patch_size[2]
    
    z_factor = int(Z/patch_z)
    x_factor = int(X/patch_x)
    y_factor = int(Y/patch_y)
    
    i_indices = list(range(int((Z-patch_z)/(patch_z/overlap_factor) + 1)))
    j_indices = list(range(int((X-patch_x)/(patch_x/overlap_factor) + 1)))
    k_indices = list(range(int((Y-patch_y)/(patch_y/overlap_factor) + 1)))
    
    z_criteria = i_indices[-1]*patch_z/overlap_factor + patch_z
    x_criteria = j_indices[-1]*patch_x/overlap_factor + patch_x
    y_criteria = k_indices[-1]*patch_y/overlap_factor + patch_y
    
    
    if z_criteria < Z:
        i_indices.append(-1)
    if x_criteria < X:
        j_indices.append(-1)
    if y_criteria < Y:
        k_indices.append(-1)    
        
    ####################### records the coordinate to segment #######################
    indices_list = []
    for i in i_indices: 
        for j in j_indices:  
            for k in k_indices:  
                if i == -1:
                    start_i = Z - patch_z
                    end_i = Z
                else:
                    start_i = int(i*patch_z/overlap_factor)
                    end_i = int(i*patch_z/overlap_factor + patch_z)
                    
                if j == -1:
                    start_j = X - patch_x
                    end_j = X
                else:
                    start_j = int(j*patch_x/overlap_factor)
                    end_j = int(j*patch_x/overlap_factor + patch_x)                    
                    
                if k == -1:
                    start_k = Y - patch_y
                    end_k = Y
                else:
                    start_k = int(k*patch_y/overlap_factor)
                    end_k = int(k*patch_y/overlap_factor + patch_y)        
        
                current_index_entry = [[start_i, end_i], [start_j, end_j], [start_k, end_k]]
        
                indices_list.append(current_index_entry)
                
    indices_list = np.array(indices_list)       
    
    if sub_vol_end_index == None:
        indices_list_to_use = indices_list[sub_vol_start_index:]
    else:
        indices_list_to_use = indices_list[sub_vol_start_index:sub_vol_end_index]
        
    ####################################################################################
        
        
        
    for ind_entry in tqdm(indices_list_to_use): 
        [[start_i, end_i], 
         [start_j, end_j], 
         [start_k, end_k]] = ind_entry
                                    
                    
                    
        img_patch = input_array[start_i : end_i,
                                start_j : end_j,
                                start_k : end_k]
                

        segmented_patch = perform_multiplanar_cube_segmentation(img_patch,
                                                                UNet_model,
                                                                patch_size = patch_size)

                    
        # drops the unreliable near boundary predictions
        if i == 0 or i == int((Z-patch_z)/(patch_z/overlap_factor)) or i == -1 or j == 0 or j == int((X-patch_x)/(patch_x/overlap_factor)) or j == -1 or k == 0 or k == int((Y-patch_y)/(patch_y/overlap_factor)) or k == -1 or overlap_factor < 3:
            drop_half_factor = 0
        else:
            drop_half_factor = 0.15

        boundary_mask = np.zeros(segmented_patch.shape)
        z_drop_half = int(segmented_patch.shape[0] * drop_half_factor)
        x_drop_half = int(segmented_patch.shape[1] * drop_half_factor)
        y_drop_half = int(segmented_patch.shape[2] * drop_half_factor)
        boundary_mask[z_drop_half : segmented_patch.shape[0] - z_drop_half,
                      x_drop_half : segmented_patch.shape[1] - x_drop_half,
                      y_drop_half : segmented_patch.shape[2] - y_drop_half] = 1
        segmented_patch = segmented_patch * boundary_mask
                
        division_tracker_patch = boundary_mask
                
                
        setpath(root_folder + results_folder + "/segmented_subvolumes")
        location_tag = str(start_i) + "_" + str(end_i) + "_" + str(start_j) + "_" + str(end_j) + "_" + str(start_k) + "_" + str(end_k)
        segmented_save_name = "segmented_" + location_tag  + ".npy"
        np.save(segmented_save_name, segmented_patch)
                
        setpath(root_folder + results_folder + "/tracker_subvolumes")
        tracker_save_name = "tracker_" + location_tag  + ".npy"
        np.save(tracker_save_name, division_tracker_patch)
                

        for l in range(len(segmented_patch)):
            # visualization
            if l < 10:
                str_name = "0000" + str(l)
            elif l < 100:
                str_name = "000" + str(l)
            elif l < 1000:
                str_name = "00" + str(l)
            elif l < 10000:
                str_name = "0" + str(l) 
            else:
                str_name = str(l)
            save_name = str_name + "_seg.png"
            save_name_img = str_name + "_img.png"
        
                
            setpath(root_folder + results_folder + "/mask_visualizations")
            cv2.imwrite(save_name, (segmented_patch[l]*255).astype(np.uint8))
            setpath(root_folder + results_folder + "/image_visualizations")
            cv2.imwrite(save_name_img, img_patch[l])
            
            
            

def merge_sub_volumes(input_array,
                      patch_size = (256, 256, 256, 3),
                      input_folder = '/inputs_folder',
                      results_folder = '/results'):
    
    """performs the segmentation"""

    # performs overlapping segmentation
    result = []
    
    Z = input_array.shape[0]
    X = input_array.shape[1]
    Y = input_array.shape[2]
    
    division_tracker = np.zeros((Z, X, Y, patch_size[3]), np.float16)
    prediction_tracker = np.zeros((Z, X, Y, patch_size[3]), np.float16)
    
    
    segmented_subvolume_names = all_file_names(root_folder + input_folder + "/segmented_subvolumes", 
                                               file_format = ".npy")
    
    
    tracker_subvolume_names = all_file_names(root_folder + input_folder + "/tracker_subvolumes", 
                                             file_format = ".npy")
    
        
    for i in tqdm(range(len(segmented_subvolume_names))): 
        
        current_mask_name = segmented_subvolume_names[i]
        current_tracker_name = tracker_subvolume_names[i]
        
        setpath(root_folder + input_folder + "/segmented_subvolumes")
        segmented_patch = np.load(current_mask_name)
        
        
        setpath(root_folder + input_folder + "/tracker_subvolumes")
        division_tracker_patch = np.load(current_tracker_name)
        
        
        location_tag = current_mask_name.split("_")
        
        start_i = int(location_tag[1])
        end_i = int(location_tag[2])
        start_j = int(location_tag[3])
        end_j = int(location_tag[4])
        start_k = int(location_tag[5])
        end_k = int(location_tag[6].split(".")[0])
        
        
        division_tracker[start_i : end_i,
                         start_j : end_j,
                         start_k : end_k] = division_tracker[start_i : end_i,
                                                             start_j : end_j,
                                                             start_k : end_k] + division_tracker_patch
        prediction_tracker[start_i : end_i,
                           start_j : end_j,
                           start_k : end_k] = prediction_tracker[start_i : end_i,
                                                                 start_j : end_j,
                                                                 start_k : end_k] + segmented_patch 
        
    
    prediction = prediction_tracker / division_tracker
    
    prediction = prediction.astype(np.float16)
    
    for l in range(len(prediction)):
        # visualization
        if l < 10:
            str_name = "0000" + str(l)
        elif l < 100:
            str_name = "000" + str(l)
        elif l < 1000:
            str_name = "00" + str(l)
        elif l < 10000:
            str_name = "0" + str(l) 
        else:
            str_name = str(l)
        save_name = str_name + "_seg.png"
        save_name_img = str_name + "_img.png"
        
        
        setpath(root_folder + results_folder + "/segmentations")
        cv2.imwrite(save_name, (prediction[l]*255).astype(np.uint8))
        setpath(root_folder + results_folder + "/images")
        cv2.imwrite(save_name_img, input_array[l])
            

    setpath(root_folder + results_folder)
    np.save("multiplanar_result.npy", prediction)



def perform_segmentation_2D_overlapping_slice(input_slice,
                                              UNet_model,
                                              overlap_factor = 3,
                                              patch_size = (256, 256, 1)):
    
    """performs the segmentation"""

    # performs overlapping segmentation

    input_array = input_slice
    
    X = input_array.shape[0]
    Y = input_array.shape[1]
    
    patch_x = patch_size[0]
    patch_y = patch_size[1]
    

    x_factor = int(X/patch_x)
    y_factor = int(Y/patch_y)
        
    
    division_tracker = np.zeros((X, Y, patch_size[2]), np.float16)
    prediction_tracker = np.zeros((X, Y, patch_size[2]), np.float16)
    

    i_indices = list(range(int((X-patch_x)/(patch_x/overlap_factor) + 1)))
    j_indices = list(range(int((Y-patch_y)/(patch_y/overlap_factor) + 1)))
    

    x_criteria = i_indices[-1]*patch_x/overlap_factor + patch_x
    y_criteria = j_indices[-1]*patch_y/overlap_factor + patch_y
    
    
    if x_criteria < X:
        i_indices.append(-1)
    if y_criteria < Y:
        j_indices.append(-1)    
        
        
    for i in tqdm(i_indices): 
        for j in j_indices:  
            if i == -1:
                start_i = X - patch_x
                end_i = X
            else:
                start_i = int(i*patch_x/overlap_factor)
                end_i = int(i*patch_x/overlap_factor + patch_x)                    
            if j == -1:
                start_j = Y - patch_y
                end_j = Y
            else:
                start_j = int(j*patch_y/overlap_factor)
                end_j = int(j*patch_y/overlap_factor + patch_y)                       
                    
                    
            img_patch = input_array[start_i : end_i,
                                    start_j : end_j]

            img_patch = img_patch/255
                

            segmented_patch = UNet_model.predict(img_patch[np.newaxis,:,:,np.newaxis],
                                                 verbose = 0)
            segmented_patch = segmented_patch[0]
            
            
            # drops the unreliable near boundary predictions
            if i == 0 or i == int((X-patch_x)/(patch_x/overlap_factor)) or i == -1 or j == 0 or j == int((Y-patch_y)/(patch_y/overlap_factor)) or j == -1 or overlap_factor < 3:
                drop_half_factor = 0
            else:
                drop_half_factor = 0.15

            boundary_mask = np.zeros(segmented_patch.shape)
            x_drop_half = int(segmented_patch.shape[0] * drop_half_factor)
            y_drop_half = int(segmented_patch.shape[1] * drop_half_factor)
            boundary_mask[x_drop_half : segmented_patch.shape[0] - x_drop_half,
                          y_drop_half : segmented_patch.shape[1] - y_drop_half] = 1
            segmented_patch = segmented_patch * boundary_mask
                
            
            division_tracker_patch = boundary_mask
                
            division_tracker[start_i : end_i,
                             start_j : end_j] = division_tracker[start_i : end_i,
                                                                 start_j : end_j] + division_tracker_patch
            prediction_tracker[start_i : end_i,
                               start_j : end_j] = prediction_tracker[start_i : end_i,
                                                                     start_j : end_j] + segmented_patch 
        
    prediction = prediction_tracker / division_tracker
    
    
    prediction = prediction.astype(np.float64)
    

    return prediction









