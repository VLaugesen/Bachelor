import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from deep_learning_support import *
import hdf5storage
import scipy
import shutil



val_indices = [8,9]#[6,7,8,9,10]
train_indices = [0,1,2,3,4,5,6,7]#[0,1,2,3,4,5]
downscale_size = (1536, 1536) #(1152, 1152) #(1536, 1536)
patch_size = (384, 384)




victor_raw_folder_img = root_folder + "/victor_data" + "/Images"
victor_raw_folder_mask = root_folder + "/victor_data" + "/Masks"
victor_raw_folder_test_img = root_folder + "/victor_data" + "/Test_Images"



img_names, img_slices = load_volume_data_slices_different_size(victor_raw_folder_img)
mask_names, mask_slices = load_volume_data_slices_different_size(victor_raw_folder_mask)

img_slices = np.array(img_slices)
mask_slices = np.array(mask_slices)


test_img_names, test_img_slices = load_volume_data_slices_different_size(victor_raw_folder_test_img)
test_img_slices = np.array(test_img_slices)



# train val split


train_img_slices = img_slices[train_indices]
train_mask_slices = mask_slices[train_indices]
val_img_slices = img_slices[val_indices]
val_mask_slices = mask_slices[val_indices]


# resizing


t_slice_img_collection = []
for t_slice_img in tqdm(train_img_slices):
    t_slice_img_collection.append(cv2.resize(t_slice_img, 
                                             dsize=downscale_size, 
                                             interpolation=cv2.INTER_AREA))
    
t_slice_mask_collection = []
for t_slice_mask in tqdm(train_mask_slices):
    t_slice_mask_collection.append(cv2.resize(t_slice_mask, 
                                              dsize=downscale_size, 
                                              interpolation=cv2.INTER_AREA))

v_slice_img_collection = []
for v_slice_img in tqdm(val_img_slices):
    v_slice_img_collection.append(cv2.resize(v_slice_img, 
                                             dsize=downscale_size, 
                                             interpolation=cv2.INTER_AREA))
    
v_slice_mask_collection = []
for v_slice_mask in tqdm(val_mask_slices):
    v_slice_mask_collection.append(cv2.resize(v_slice_mask, 
                                              dsize=downscale_size, 
                                              interpolation=cv2.INTER_AREA))
    
    
train_img_slices = np.array(t_slice_img_collection)
train_mask_slices = np.array(t_slice_mask_collection)
val_img_slices = np.array(v_slice_img_collection)
val_mask_slices = np.array(v_slice_mask_collection)    
    




test_downscale_size = (int(np.round(test_img_slices.shape[1]/(5000/downscale_size[0]))), 
                       int(np.round(test_img_slices.shape[1]/(5000/downscale_size[0]))))

test_slice_img_collection = []
for test_slice_img in tqdm(test_img_slices):
    test_slice_img_collection.append(cv2.resize(test_slice_img, 
                                                dsize=test_downscale_size, 
                                                interpolation=cv2.INTER_AREA))

test_img_slices = np.array(test_slice_img_collection)




# Cutting
cut_images_into_patches_2D_stack_train_val_split(train_img_slices,
                                                 raw_folder_train_img,
                                                 patch_size,
                                                 patch_step_ratio_v = 2,
                                                 patch_step_ratio_h = 2,
                                                 intial_tag = "train_img_")

cut_images_into_patches_2D_stack_train_val_split(train_mask_slices,
                                                 raw_folder_train_mask,
                                                 patch_size,
                                                 patch_step_ratio_v = 2,
                                                 patch_step_ratio_h = 2,
                                                 intial_tag = "train_mask_")

cut_images_into_patches_2D_stack_train_val_split(val_img_slices,
                                                 raw_folder_valid_img,
                                                 patch_size,
                                                 patch_step_ratio_v = 2,
                                                 patch_step_ratio_h = 2,
                                                 intial_tag = "val_img_")

cut_images_into_patches_2D_stack_train_val_split(val_mask_slices,
                                                 raw_folder_valid_mask,
                                                 patch_size,
                                                 patch_step_ratio_v = 2,
                                                 patch_step_ratio_h = 2,
                                                 intial_tag = "val_mask_")


save_img_as_slices_with_names(test_img_slices,
                              test_folder,
                              test_img_names)
