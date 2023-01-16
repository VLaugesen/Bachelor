# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 03:49:21 2021

@author: CWANG
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from deep_learning_support import *
import hdf5storage
import scipy
import shutil
import torch

# todos
if_train = True

if_use_pretrained_weights = True

if_segment = True



# params
batch_size = 4
epoch = 1000
Patch_Size = (384, 384)




# model save names
model_pretrained_name = "unet_2D_mito.hdf5" 
model_save_name = "unet_2D_axon.hdf5"





# training
if if_train == True:
   
    train_names = np.array(all_file_names(raw_folder_train_img))
    valid_names = np.array(all_file_names(raw_folder_valid_img))
    
    train_gen = prepare_training_generator(batch_size = batch_size,
                                           epochs = epoch,
                                           input_img_path = raw_folder_train_img,
                                           input_mask_path = raw_folder_train_mask,
                                           patch_size = Patch_Size)
    
    valid_gen = prepare_validation_generator(batch_size = batch_size,
                                             epochs = epoch,
                                             input_img_path = raw_folder_valid_img,
                                             input_mask_path = raw_folder_valid_mask,
                                             patch_size = Patch_Size)



    model_save_path = root_folder + "\\" + model_save_name
    
    UNet_model = unet(input_shape = (Patch_Size[0], 
                                     Patch_Size[1], 
                                     1), 
                      loss_func = IOU_loss,
                      metric_func = 'accuracy',
                      learning_rate = 0.0001,
                      feed_weights = False)
    
    
    
    if if_use_pretrained_weights == True: 
        setpath(root_folder)

        # loads the pretrained weights
        UNet_weights_model = tf.keras.models.load_model(model_pretrained_name, 
                                                        custom_objects={'IOU_loss': IOU_loss})
    
    
        # sets the weights with the exception of the last layer
        for index in range(len(UNet_weights_model.layers)):
            UNet_model.layers[index].set_weights(UNet_weights_model.layers[index].get_weights())
        
        
    


    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path, 
                                                    monitor = "val_loss", 
                                                    verbose = 0, 
                                                    save_best_only = True, 
                                                    mode='min')
    
    
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                     factor = 0.2,
                                                     patience = 3,
                                                     min_lr= 0.0000001)
    
    
    
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                          mode = 'min', 
                                          verbose = 1, 
                                          patience = 10)
    


    history = UNet_model.fit_generator(train_gen, 
                                       steps_per_epoch = len(train_names) // batch_size, 
                                       epochs = epoch, 
                                       verbose = 1, 
                                       callbacks=[checkpoint, reduce_lr, es],
                                       validation_data = valid_gen,
                                       validation_steps = len(valid_names) // batch_size)
    


    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    print()







if if_segment == True:

    setpath(test_folder)
    print()

    # loads images, normalizes intensities between 0 and 1, and reshapes into UNet shape.
    test_names, test_imgs = load_volume_data_slices(test_folder)
  
    
    # loads model
    setpath(root_folder)
    UNet_model = tf.keras.models.load_model(model_save_name, 
                                            custom_objects={'IOU_loss': IOU_loss})
    
    
    # performs segmentation
    test_results = []
    print()
    for test_slice in test_imgs:
        print("- segmenting img in patches...")
        slice_result = perform_segmentation_2D_overlapping_slice(test_slice,
                                                                 UNet_model,
                                                                 overlap_factor = 4,
                                                                 patch_size = (Patch_Size[0], 
                                                                               Patch_Size[1], 
                                                                               1))
        test_results.append(slice_result)
        print()

    test_results = np.array(test_results)




    # saves results
    save_img_as_slices_with_names(test_imgs,
                                  root_folder + "/output" + "/images",
                                  test_names)
    
    
    save_img_as_slices_with_names((test_results * 255).astype(np.uint8),
                                  root_folder + "/output" + "/masks",
                                  test_names)











    
 