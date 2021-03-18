#!/usr/bin/env python
# coding: utf-8

# # Prepare SpaceNet 7 Data for Model Training
# 
# We assume that initial steps of README have been executed and that this notebook is running in a docker container.  See the `src` directory for functions used in the algorithm.  

# In[2]:


# Dataset location (edit as needed)
root_dir = '/app/spacenet7/'


# In[3]:


import multiprocessing
import pandas as pd
import numpy as np
import skimage
import gdal
import sys
import os

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 16})
mpl.rcParams['figure.dpi'] = 300

import solaris as sol
from solaris.raster.image import create_multiband_geotiff
from solaris.utils.core import _check_gdf_load

# import from data_prep_funcs
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from sn7_baseline_prep_funcs import map_wrapper, make_geojsons_and_masks


# In[4]:


# Create Training Masks
# Multi-thread to increase speed
# We'll only make a 1-channel mask for now, but Solaris supports a multi-channel mask as well, see
#     https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb

aois = sorted([f for f in os.listdir(os.path.join(root_dir, 'train'))
               if os.path.isdir(os.path.join(root_dir, 'train', f))])
n_threads = 10
params = [] 
make_fbc = False

input_args = []
for i, aoi in enumerate(aois):
    print(i, "aoi:", aoi)
    im_dir = os.path.join(root_dir, 'train', aoi, 'images_masked/')
    json_dir = os.path.join(root_dir, 'train', aoi, 'labels_match/')
    out_dir_mask = os.path.join(root_dir, 'train', aoi, 'masks/')
    out_dir_mask_fbc = os.path.join(root_dir, 'train', aoi, 'masks_fbc/')
    os.makedirs(out_dir_mask, exist_ok=True)
    if make_fbc:
        os.makedirs(out_dir_mask_fbc, exist_ok=True)

    json_files = sorted([f
                for f in os.listdir(os.path.join(json_dir))
                if f.endswith('Buildings.geojson') and os.path.exists(os.path.join(json_dir, f))])
    for j, f in enumerate(json_files):
        # print(i, j, f)
        name_root = f.split('.')[0]
        json_path = os.path.join(json_dir, f)
        image_path = os.path.join(im_dir, name_root + '.tif').replace('labels', 'images').replace('_Buildings', '')
        output_path_mask = os.path.join(out_dir_mask, name_root + '.tif')
        if make_fbc:
            output_path_mask_fbc = os.path.join(out_dir_mask_fbc, name_root + '.tif')
        else:
            output_path_mask_fbc = None
            
        if (os.path.exists(output_path_mask)):
             continue
        else: 
            input_args.append([make_geojsons_and_masks, 
                               name_root, image_path, json_path,
                               output_path_mask, output_path_mask_fbc])

# execute 
print("len input_args", len(input_args))
print("Execute...\n")
with multiprocessing.Pool(n_threads) as pool:
    pool.map(map_wrapper, input_args[:10])


# In[50]:


import cv2
plot = False
for aoi in aois:
    out_dir_cd_map = os.path.join(root_dir, 'train', aoi, 'cd_maps/')
    if not os.path.exists(out_dir_cd_map):
        os.mkdir(out_dir_cd_map)
    mask_dir = os.path.join(root_dir, 'train', aoi, 'masks/')
    mask_files = sorted([f
                for f in os.listdir(os.path.join(mask_dir))
                if os.path.exists(os.path.join(mask_dir, f))])
    udm_dir = os.path.join(root_dir, 'train', aoi, 'UDM_masks/')
    for f1, f2 in zip(mask_files, mask_files[1:]):
        im1 = cv2.imread(os.path.join(mask_dir, f1), cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(os.path.join(mask_dir, f2), cv2.IMREAD_GRAYSCALE)
        cm = cv2.bitwise_xor(im1, im2)
        for filename in (f1, f2):
            udm_path = os.path.join(udm_dir, filename.replace('Buildings', 'UDM'))
            if os.path.exists(udm_path):
                udm = cv2.imread(udm_path, cv2.IMREAD_GRAYSCALE)
                cm = cv2.bitwise_and(cm, cv2.bitwise_not(udm))
        if plot:
            figsize=(24, 24)
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=figsize)
            _ = ax0.imshow(im1)
            ax0.set_xticks([])
            ax0.set_yticks([])
            # _ = ax0.set_title(name)
            _ = ax1.imshow(im2)
            ax1.set_xticks([])
            ax1.set_yticks([])
            # _ = ax1.set_title(name)
            _ = ax2.imshow(mask)
            ax2.set_xticks([])
            ax2.set_yticks([])
            # _ = ax2.set_title(name)
            _ = ax3.imshow(cm)
            ax3.set_xticks([])
            ax3.set_yticks([])
            # _ = ax2.set_title(name)
            _ = fig.suptitle(name)
            plt.tight_layout()
        date2 = f2.split('.')[0].split('global_monthly_')[-1]
        date1 = f1.split('.')[0].split('global_monthly_')[-1][:7]
        cd_name = f'global_monthly_{date1}-{date2}.tif'
        output_path_mask = os.path.join(out_dir_cd_map, cd_name)
        cv2.imwrite(output_path_mask, cm)


# In[4]:


# Inspect visually

aoi = 'L15-0331E-1257N_1327_3160_13'
im_dir = os.path.join(root_dir, 'train', aoi, 'images_masked')
mask_dir = os.path.join(root_dir, 'train', aoi, 'masks')

im_list = sorted([z for z in os.listdir(im_dir) if z.endswith('.tif')])
im_file = im_list[0]

im_path = os.path.join(im_dir, im_file)
mask_path = os.path.join(mask_dir, im_file.split('.')[0] + '_Buildings.tif')
im = skimage.io.imread(im_path)
mask = skimage.io.imread(mask_path)

figsize=(24, 12)
name = im_file.split('.')[0].split('global_monthly_')[-1]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
_ = ax0.imshow(im)
ax0.set_xticks([])
ax0.set_yticks([])
# _ = ax0.set_title(name)
_ = ax1.imshow(mask)
ax1.set_xticks([])
ax1.set_yticks([])
# _ = ax1.set_title(name)
_ = fig.suptitle(name)
plt.tight_layout()


# In[6]:


# Make dataframe csvs for train/test

out_dir = os.path.join(root_dir, 'csvs/')
pops = ['train', 'test_public']
os.makedirs(out_dir, exist_ok=True)

for pop in pops: 
    d = os.path.join(root_dir, pop)
    outpath = os.path.join(out_dir, 'sn7_baseline_' + pop + '_df.csv')
    im_list, mask_list = [], []
    subdirs = sorted([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))])
    for subdir in subdirs:
        
        if pop == 'train':
            im_files = [os.path.join(d, subdir, 'images_masked', f)
                    for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                    if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
            mask_files = [os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif')
                      for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                      if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
            im_list.extend(im_files)
            mask_list.extend(mask_files)
    
        elif pop == 'test_public':
            im_files = [os.path.join(d, subdir, 'images_masked', f)
                    for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                    if f.endswith('.tif')]
            im_list.extend(im_files)

    # save to dataframes
    # print("im_list:", im_list)
    # print("mask_list:", mask_list)
    if pop == 'train':
        df = pd.DataFrame({'image': im_list, 'label': mask_list})
        # display(df.head())
    elif pop == 'test_public':
        df = pd.DataFrame({'image': im_list})
    df.to_csv(outpath, index=False)
    print(pop, "len df:", len(df))
    print("output csv:", outpath)


# --------
# We are now ready to proceed with training and testing, see sn7_baseline.ipynb.

# In[ ]:




