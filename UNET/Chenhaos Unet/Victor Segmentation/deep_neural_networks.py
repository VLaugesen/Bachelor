# -*- coding: utf-8 -*-
"""
Various Convolutional Neural Networks for Image Segmentation

"""
import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops


####################################################################################################################
# Support Functions

dice_smooth = 0.1
IOU_smooth = 0.1



def multiclass_IOU_score(y_true, y_pred):
    """
    standard multiclass IOU score (mean)
    """
    intersection = tf.keras.backend.sum(y_true * y_pred, axis = [0,1,2])
    union = (tf.keras.backend.sum(y_true, axis = [0,1,2]) +
             tf.keras.backend.sum(y_pred, axis = [0,1,2]) -
             intersection)
    scores = (intersection + IOU_smooth)/(union + IOU_smooth)

    return tf.keras.backend.mean(scores)


def multiclass_IOU_loss(y_true, y_pred):
    """
    standard multiclass IOU loss (mean) for image segmentation
    """
    
    return 1 - multiclass_IOU_score(y_true, y_pred)


def dice_score(y_true, y_pred):
    """
    standard dice score
    """
    intersection = tf.keras.backend.sum(y_true * y_pred)
    divisor = (tf.keras.backend.sum(y_true) +
               tf.keras.backend.sum(y_pred))
    return (2 * intersection + dice_smooth) / (divisor + dice_smooth)


def dice_loss(y_true, y_pred):
    """
    standard dice loss for image segmentation
    """
    return 1 - dice_score(y_true, y_pred)
    

def custom_dice_score(weights):
    """
    custom weighted dice score
    """
    def weighted_dice_score(y_true, y_pred):
        intersection = tf.keras.backend.sum(y_true * y_pred)
        divisor = (tf.keras.backend.sum(y_true) +
                   tf.keras.backend.sum(tf.math.multiply(y_pred, weights)))
        return (2 * intersection + dice_smooth) / (divisor + dice_smooth)
    return weighted_dice_score


def custom_dice_loss(weights):
    """
    custom weighted dice loss for image segmentation
    """
    def weighted_dice_loss(y_true, y_pred):
        intersection = tf.keras.backend.sum(y_true * y_pred)
        divisor = (tf.keras.backend.sum(y_true) +
                   tf.keras.backend.sum(tf.math.multiply(y_pred, weights)))
        dice_coeffient = (2 * intersection + dice_smooth) / (divisor + dice_smooth)
        return 1 - dice_coeffient
    return weighted_dice_loss



def IOU_score(y_true, y_pred):
    """
    standard IOU score
    """
    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = (tf.keras.backend.sum(y_true) +
             tf.keras.backend.sum(y_pred) -
             intersection)
    return (intersection + IOU_smooth)/(union + IOU_smooth)


def IOU_loss(y_true, y_pred):
    """
    standard IOU loss for image segmentation
    """
    return 1 - IOU_score(y_true, y_pred)
    

def custom_IOU_score(weights):
    """
    custom weighted IOU score
    """
    def weighted_IOU_score(y_true, y_pred):
        intersection = tf.keras.backend.sum(y_true * y_pred)
        union = (tf.keras.backend.sum(y_true) +
                 tf.keras.backend.sum(tf.math.multiply(y_pred, weights)) -
                 intersection)
        return (intersection + IOU_smooth)/(union + IOU_smooth)
    return weighted_IOU_score


def custom_IOU_loss(weights):
    """
    custom weighted IOU loss for image segmentation
    """
    def weighted_IOU_loss(y_true, y_pred):
        intersection = tf.keras.backend.sum(y_true * y_pred)
        union = (tf.keras.backend.sum(y_true) +
                 tf.keras.backend.sum(tf.math.multiply(y_pred, weights)) -
                 intersection)
        IOU_coefficient = (intersection + IOU_smooth)/(union + IOU_smooth)
        return 1 - IOU_coefficient
    return weighted_IOU_loss




####################################################################################################################
# 2D Unet

def unet(input_shape = (192,192,1), 
         loss_func = "binary_crossentropy",
         metric_func = 'accuracy',
         learning_rate = 0.00001,
         feed_weights = False):
    """
    U-Net: Convolutional Networks for BiomedicalImage Segmentation
    https://arxiv.org/pdf/1505.04597.pdf
    """

    k_size = 3
    dilation = 1
    up_dilation = 1
    kernel_choice = 'he_normal'

    Input_Layer = tf.keras.layers.Input(shape = input_shape)
    
    if feed_weights == True:
        Weight_Layer = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 1))
    
    conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(Input_Layer)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_1)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    down_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1)

    
    conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_2)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    down_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)

    
    conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_3)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    down_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_3)

    
    conv_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_4)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    down_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_4)


    conv_5 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_4)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_5)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    up_1 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_5) 
    up_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_1)
    
    
    merg_1 = tf.keras.layers.concatenate([conv_4, up_1], axis = 3)
    merg_1 = tf.keras.layers.Dropout(0.5)(merg_1)
    conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_1)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_6)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    up_2 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_6)
    up_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_2)
    
    
    merg_2 = tf.keras.layers.concatenate([conv_3,up_2], axis = 3)
    merg_2 = tf.keras.layers.Dropout(0.5)(merg_2)
    conv_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_2)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_7)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    up_3 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_7)
    up_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_3)
    
    
    merg_3 = tf.keras.layers.concatenate([conv_2,up_3], axis = 3)
    merg_3 = tf.keras.layers.Dropout(0.5)(merg_3)
    conv_8 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_3)
    conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
    conv_8 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_8)
    conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
    up_4 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_8)
    up_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_4)
    
    
    merg_4 = tf.keras.layers.concatenate([conv_1,up_4], axis = 3)
    merg_4 = tf.keras.layers.Dropout(0.5)(merg_4)
    conv_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_4)
    conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
    conv_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_9)    
    conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
    
    
    out_conv = tf.keras.layers.Conv2D(filters=2, kernel_size=(k_size,k_size), dilation_rate=(1, 1), kernel_initializer= kernel_choice, 
                                    activation = 'relu', padding = 'same')(conv_9)
    Output_Layer = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(out_conv)

    if feed_weights == True:
        model = tf.keras.models.Model(inputs = [Input_Layer, Weight_Layer], outputs = Output_Layer)
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), 
                      loss = loss_func(Weight_Layer), metrics = [metric_func])
        model.summary()   
    else:
        model = tf.keras.models.Model(inputs = Input_Layer, outputs = Output_Layer)
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), 
                      loss = loss_func, metrics = [metric_func])
        model.summary()   
    
    return model





def transfer_UNET(input_shape = (384,576,3),  
                  loss_func = "binary_crossentropy",
                  metric_func = 'accuracy',
                  feed_weights = True):
    
    """UNET with VGG16 encoder to allow for transfer learning"""
    
    # Downward Path
    downward_path = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape = input_shape)
    downward_path_corrected = tf.keras.Model(inputs=downward_path.inputs, outputs=downward_path.layers[-2].output)
    
    if feed_weights == True:
        Weight_Layer = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 1))
    
    Input_Layer = downward_path_corrected.layers[0].output
    conv_block_1 = downward_path_corrected.layers[2].output
    conv_block_2 = downward_path_corrected.layers[5].output
    conv_block_3 = downward_path_corrected.layers[9].output
    conv_block_4 = downward_path_corrected.layers[13].output
    conv_block_5 = downward_path_corrected.layers[17].output
    
    
    # Upward Path
    norm_block_5 = tf.keras.layers.BatchNormalization()(conv_block_5)
    up_block_5 = tf.keras.layers.UpSampling2D(size = (2,2))(norm_block_5) 
    up_block_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(up_block_5)
    
    
    norm_block_4 = tf.keras.layers.BatchNormalization()(conv_block_4)
    #norm_block_4 = tf.keras.layers.Cropping2D(cropping = 6)(norm_block_4)
    conv_block_6 = tf.keras.layers.concatenate([norm_block_4, up_block_5], axis = 3)
    conv_block_6 = tf.keras.layers.Dropout(0.5)(conv_block_6)
    conv_block_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_6)
    conv_block_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_6)
    up_block_6 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_block_6)
    up_block_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(up_block_6)


    norm_block_3 = tf.keras.layers.BatchNormalization()(conv_block_3)
    #norm_block_3 = tf.keras.layers.Cropping2D(cropping = 6)(norm_block_3)
    conv_block_7 = tf.keras.layers.concatenate([norm_block_3, up_block_6], axis = 3)
    conv_block_7 = tf.keras.layers.Dropout(0.5)(conv_block_7)
    conv_block_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_7)
    conv_block_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_7)
    up_block_7 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_block_7)
    up_block_7 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(up_block_7)


    norm_block_2 = tf.keras.layers.BatchNormalization()(conv_block_2)
    #norm_block_2 = tf.keras.layers.Cropping2D(cropping = 6)(norm_block_2)
    conv_block_8 = tf.keras.layers.concatenate([norm_block_2, up_block_7], axis = 3)
    conv_block_8 = tf.keras.layers.Dropout(0.5)(conv_block_8)
    conv_block_8 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_8)
    conv_block_8 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_8)
    up_block_8 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_block_8)
    up_block_8 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(up_block_8)


    norm_block_1 = tf.keras.layers.BatchNormalization()(conv_block_1)
    #norm_block_1 = tf.keras.layers.Cropping2D(cropping = 4)(norm_block_1)
    conv_block_9 = tf.keras.layers.concatenate([norm_block_1, up_block_8], axis = 3)
    conv_block_9 = tf.keras.layers.Dropout(0.5)(conv_block_9)
    conv_block_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_9)
    conv_block_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_9)
    

    
    conv_one_hot = tf.keras.layers.Conv2D(filters=2, kernel_size=(3,3), kernel_initializer= "he_normal", activation = 'relu', padding = 'same')(conv_block_9)    
    Output_Layer = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv_one_hot)
    


    if feed_weights == True:
        model = tf.keras.models.Model(inputs = [Input_Layer, Weight_Layer], outputs = Output_Layer)
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.00001), 
                      loss = loss_func(Weight_Layer), metrics = [metric_func])
        model.summary()   
        
    else:
        model = tf.keras.models.Model(inputs = Input_Layer, outputs = Output_Layer)
        model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.00001), 
                      loss = loss_func, metrics = [metric_func])
        model.summary()   

    return model



# multi-class unet

def multiclass_unet(input_shape = (192,192,1), 
                    loss_func = "binary_crossentropy",
                    metric_func = 'accuracy',
                    learning_rate = 0.00001):
    """
    U-Net: Convolutional Networks for BiomedicalImage Segmentation
    https://arxiv.org/pdf/1505.04597.pdf
    """

    k_size = 3
    dilation = 1
    up_dilation = 1
    kernel_choice = 'he_normal'

    Input_Layer = tf.keras.layers.Input(shape = input_shape)
    
    
    conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(Input_Layer)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_1)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    down_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1)

    
    conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_2)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    down_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)

    
    conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_3)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    down_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_3)

    
    conv_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(conv_4)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    down_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_4)


    conv_5 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(k_size,k_size), 
                                    dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                    activation = 'relu', padding = 'same')(down_4)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_5)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    up_1 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_5) 
    up_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_1)
    
    
    merg_1 = tf.keras.layers.concatenate([conv_4, up_1], axis = 3)
    merg_1 = tf.keras.layers.Dropout(0.5)(merg_1)
    conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_1)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_6)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    up_2 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_6)
    up_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_2)
    
    
    merg_2 = tf.keras.layers.concatenate([conv_3,up_2], axis = 3)
    merg_2 = tf.keras.layers.Dropout(0.5)(merg_2)
    conv_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_2)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Conv2D(filters=256, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_7)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    up_3 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_7)
    up_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_3)
    
    
    merg_3 = tf.keras.layers.concatenate([conv_2,up_3], axis = 3)
    merg_3 = tf.keras.layers.Dropout(0.5)(merg_3)
    conv_8 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_3)
    conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
    conv_8 = tf.keras.layers.Conv2D(filters=128, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_8)
    conv_8 = tf.keras.layers.BatchNormalization()(conv_8)
    up_4 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_8)
    up_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), 
                                     dilation_rate=(up_dilation, up_dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(up_4)
    
    
    merg_4 = tf.keras.layers.concatenate([conv_1,up_4], axis = 3)
    merg_4 = tf.keras.layers.Dropout(0.5)(merg_4)
    conv_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(merg_4)
    conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
    conv_9 = tf.keras.layers.Conv2D(filters=64, kernel_size=(k_size,k_size), 
                                     dilation_rate=(dilation, dilation), kernel_initializer= kernel_choice,
                                     activation = 'relu', padding = 'same')(conv_9)    
    conv_9 = tf.keras.layers.BatchNormalization()(conv_9)
    
    
    Output_Layer = tf.keras.layers.Conv2D(filters=3, 
                                          kernel_size=1, 
                                          dilation_rate=(1, 1), 
                                          activation = 'softmax', 
                                          padding = 'same')(conv_9)
    
    
    model = tf.keras.models.Model(inputs = Input_Layer, outputs = Output_Layer)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = learning_rate), 
                  loss = loss_func, metrics = [metric_func])
    
    
    model.summary()   
    
    return model




####################################################################################################################
# 3D Unet

def volumetric_unet(input_shape = (192,192,192,1),
                    loss_func = "binary_crossentropy",
                    metric_func = 'accuracy'):
    """
    3D U-Net: Learning Dense VolumetricSegmentation from Sparse Annotation
    https://arxiv.org/pdf/1606.06650.pdf
    """
    
    Input_Layer = tf.keras.layers.Input(shape=input_shape)
    
    
    # Downward path
    conv_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(Input_Layer)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_1) 
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    down_1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_1)
    

    conv_2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(down_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_2)    
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    down_2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_2)    
    

    conv_3 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(down_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_3)    
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    down_3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_3)    

    
    conv_4 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(down_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Conv3D(filters=512, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_4)    
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    up_1 = tf.keras.layers.UpSampling3D(size = (2,2,2))(conv_4)     
    up_1 = tf.keras.layers.Conv3D(filters=512, kernel_size=(2,2,2), kernel_initializer= "he_normal",
                                  activation = 'relu', padding = 'same')(up_1)
    
    
    # Upward path
    
    merg_1 = tf.keras.layers.concatenate([conv_3,up_1], axis = 4)
    conv_5 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(merg_1)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_5)    
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    up_2 = tf.keras.layers.UpSampling3D(size = (2,2,2))(conv_5)     
    up_2 = tf.keras.layers.Conv3D(filters=256, kernel_size=(2,2,2), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(up_2)   
    

    merg_2 = tf.keras.layers.concatenate([conv_2,up_2], axis = 4)
    conv_6 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(merg_2)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_6)    
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    up_3 = tf.keras.layers.UpSampling3D(size = (2,2,2))(conv_6)     
    up_3 = tf.keras.layers.Conv3D(filters=128, kernel_size=(2,2,2), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(up_3)       
    
    
    merg_3 = tf.keras.layers.concatenate([conv_1,up_3], axis = 4)
    conv_7 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(merg_3)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_7)    
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    
    
    # Results
    out_conv = tf.keras.layers.Conv3D(filters=2, kernel_size=(1,1,1), kernel_initializer= "he_normal",
                                      activation = 'relu', padding = 'same')(conv_7)
    Output_Layer = tf.keras.layers.Conv3D(1, 1, activation = 'sigmoid')(out_conv)


    # Setting up and compiling the model
    model = tf.keras.models.Model(inputs = Input_Layer, outputs = Output_Layer)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.00001), 
                  loss = loss_func, metrics = [metric_func])
   
    model.summary()
    
    return model


def multiclass_volumetric_unet(input_shape = (192,192,192,1),
                               loss_func = "binary_crossentropy",
                               metric_func = 'accuracy'):
    """
    3D U-Net: Learning Dense VolumetricSegmentation from Sparse Annotation
    https://arxiv.org/pdf/1606.06650.pdf
    """
    
    Input_Layer = tf.keras.layers.Input(shape=input_shape)
    
    
    # Downward path
    conv_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(Input_Layer)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_1) 
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    down_1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_1)
    

    conv_2 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(down_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_2)    
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    down_2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_2)    
    

    conv_3 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(down_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_3)    
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    down_3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv_3)    

    
    conv_4 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(down_3)
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    conv_4 = tf.keras.layers.Conv3D(filters=512, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_4)    
    conv_4 = tf.keras.layers.BatchNormalization()(conv_4)
    up_1 = tf.keras.layers.UpSampling3D(size = (2,2,2))(conv_4)     
    up_1 = tf.keras.layers.Conv3D(filters=512, kernel_size=(2,2,2), kernel_initializer= "he_normal",
                                  activation = 'relu', padding = 'same')(up_1)
    
    
    # Upward path
    
    merg_1 = tf.keras.layers.concatenate([conv_3,up_1], axis = 4)
    conv_5 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(merg_1)
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    conv_5 = tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_5)    
    conv_5 = tf.keras.layers.BatchNormalization()(conv_5)
    up_2 = tf.keras.layers.UpSampling3D(size = (2,2,2))(conv_5)     
    up_2 = tf.keras.layers.Conv3D(filters=256, kernel_size=(2,2,2), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(up_2)   
    

    merg_2 = tf.keras.layers.concatenate([conv_2,up_2], axis = 4)
    conv_6 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(merg_2)
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    conv_6 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_6)    
    conv_6 = tf.keras.layers.BatchNormalization()(conv_6)
    up_3 = tf.keras.layers.UpSampling3D(size = (2,2,2))(conv_6)     
    up_3 = tf.keras.layers.Conv3D(filters=128, kernel_size=(2,2,2), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(up_3)       
    
    
    merg_3 = tf.keras.layers.concatenate([conv_1,up_3], axis = 4)
    conv_7 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(merg_3)
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    conv_7 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,3), kernel_initializer= "he_normal",
                                    activation = 'relu', padding = 'same')(conv_7)    
    conv_7 = tf.keras.layers.BatchNormalization()(conv_7)
    
    
    # Results
    Output_Layer = tf.keras.layers.Conv3D(3, 1, activation = 'softmax')(conv_7)



    # Setting up and compiling the model
    model = tf.keras.models.Model(inputs = Input_Layer, outputs = Output_Layer)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.00001), 
                  loss = loss_func, metrics = [metric_func])
   
    model.summary()
    
    return model


    