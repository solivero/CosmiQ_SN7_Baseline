#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:11:02 2020

@author: avanetten
"""

from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing
import pandas as pd
import numpy as np
import skimage.io as ski
import tqdm
import glob
import math
import gdal
import time
import os
import cv2

import solaris as sol
from solaris.utils.core import _check_gdf_load
from solaris.raster.image import create_multiband_geotiff 

def score(pred_cm_dir, label_cm_dir):
    pred_cm_list = sorted([z for z in os.listdir(pred_cm_dir) if z.endswith('.tif')])
    def score_file(pred_cm_file):
        pred_im = cv2.imread(os.path.join(pred_cm_dir, pred_cm_file), cv2.IMREAD_GRAYSCALE)
        label_im = cv2.imread(os.path.join(label_cm_dir, pred_cm_file), cv2.IMREAD_GRAYSCALE)
        print(os.path.exists(os.path.join(pred_cm_dir, pred_cm_file)), os.path.exists(os.path.join(label_cm_dir, pred_cm_file)))
        label_im_cat = np.zeros(pred_im.shape, dtype=np.int8)
        label_im_cat[label_im == 255] = 1
        pred_im_cat = np.zeros(label_im.shape)
        pred_im_cat[pred_im == 255] = 1
        return sol.eval.pixel.f1(label_im_cat, pred_im_cat, verbose=True)
    scores = np.array([score_file(pred_cm_file) for pred_cm_file in pred_cm_list])
    print(scores.shape)
    print(np.mean(scores, axis=0))
    return scores

def group_pred(pred_top_dir):
    raw_name = 'raw/'
    grouped_name = 'grouped/'
    im_list = sorted([z for z in os.listdir(os.path.join(pred_top_dir, raw_name)) if z.endswith('.tif')])
    df = pd.DataFrame({'image': im_list})
    roots = [z.split('mosaic_')[-1].split('.tif')[0] for z in df['image'].values]
    df['root'] = roots
    # copy files
    for idx, row in df.iterrows():
        in_path_tmp = os.path.join(pred_top_dir, raw_name, row['image'])
        out_dir_tmp = os.path.join(pred_top_dir, grouped_name, row['root'], 'masks')
        os.makedirs(out_dir_tmp, exist_ok=True)
        cmd = 'cp ' + in_path_tmp + ' ' + out_dir_tmp
        print("cmd:", cmd)
        os.system(cmd)   

def change_map_from_masks(mask_dir, cm_out_dir, udm_dir=None, months=1):
    if not os.path.exists(cm_out_dir):
        os.mkdir(cm_out_dir)
    mask_files = sorted([f
                for f in os.listdir(os.path.join(mask_dir))
                if os.path.exists(os.path.join(mask_dir, f))])
    print(mask_files)
    first = True
    for f1, f2 in zip(mask_files, mask_files[months:]):
        im1 = np.array(cv2.imread(os.path.join(mask_dir, f1), cv2.IMREAD_LOAD_GDAL), dtype=np.uint8)
        im2 = np.array(cv2.imread(os.path.join(mask_dir, f2), cv2.IMREAD_LOAD_GDAL), dtype=np.uint8)
        #cm = cv2.bitwise_xor(im1, im2)
        #cm = cv2.subtract(im2, im1)
        cm = np.abs(im2.astype(np.int8) - im1.astype(np.int8)) # 255 -> -1 in type conversion.
        changes = cm == 1
        cm = np.zeros(im1.shape, dtype=np.uint8)
        cm[changes] = 255
        if first:
            print(im1)
            print(im2)
            print(cm)
            first = False
        if udm_dir:
            for filename in (f1, f2):
                udm_path = os.path.join(udm_dir, filename.replace('Buildings', 'UDM'))
                if os.path.exists(udm_path):
                    udm = cv2.imread(udm_path, cv2.IMREAD_GRAYSCALE)
                    cm = cv2.bitwise_and(cm, cv2.bitwise_not(udm))
        date2 = f2.split('.')[0].split('global_monthly_')[-1]
        date1 = f1.split('.')[0].split('global_monthly_')[-1][:7]
        cm_name = f'global_monthly_{date1}-{date2}.tif'.replace('_Buildings', '')
        output_path_mask = os.path.join(cm_out_dir, cm_name)
        cv2.imwrite(output_path_mask, cm)

def map_wrapper(x):
    '''For multi-threading'''
    return x[0](*(x[1:]))
    
    
def multithread_polys(param):
    '''Simple wrapper around mask_to_poly_geojson() for multiprocessing
    # https://solaris.readthedocs.io/en/latest/_modules/solaris/vector/mask.html#mask_to_poly_geojson
    # mask_to_poly_geojson(pred_arr, channel_scaling=None, reference_im=None,
    #                          output_path=None, output_type='geojson', min_area=40,
    #                          bg_threshold=0, do_transform=None, simplify=False,
    #                          tolerance=0.5, **kwargs)
    '''
    
    [pred_image, min_area, output_path_pred, output_type, 
             bg_threshold, simplify] = param
    print("output_pred:", os.path.basename(output_path_pred))
    sol.vector.mask.mask_to_poly_geojson(pred_image, 
                                         min_area=min_area, 
                                         output_path=output_path_pred,
                                         output_type=output_type,
                                         bg_threshold=bg_threshold,
                                         simplify=simplify)
                                         
    
def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.
    Adapted from: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/iou.py, but
    keeps index of test_data_GDF
    
    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.
    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``test_data_GDF`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.
    """

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for idx, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
            gt_idx = idx
        else:
            iou_score = 0
            gt_idx = -1
        row['iou_score'] = iou_score
        row['gt_idx'] = gt_idx
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF

    
def track_footprint_identifiers(json_dir, out_dir,
                          min_iou=0.25, iou_field='iou_score', id_field='Id',
                          reverse_order=False,     
                          verbose=True, super_verbose=False):
    '''
    Track footprint identifiers in the deep time stack.
    We need to track the global gdf instead of just the gdf of t-1.
    '''
        
    os.makedirs(out_dir, exist_ok=True)
    
    # set columns for master gdf
    gdf_master_columns = [id_field, iou_field, 'area', 'geometry']

    json_files = sorted([f
                for f in os.listdir(os.path.join(json_dir))
                if f.endswith('.geojson') and os.path.exists(os.path.join(json_dir, f))])
    # start at the end and work backwards?
    if reverse_order:
        json_files = json_files[::-1]

    # check if only partical matching has been done (this will cause errors)
    out_files_tmp = sorted([z for z in os.listdir(out_dir) if z.endswith('.geojson')])
    if len(out_files_tmp) > 0:
        if len(out_files_tmp) != len(json_files):
            raise Exception("\nError in:", out_dir, "with N =", len(out_files_tmp), 
                            "files, need to purge this folder and restart matching!\n")
            return
        elif len(out_files_tmp) == len(json_files):
            print("\nDir:", os.path.basename(out_dir), "N files:", len(json_files), 
                  "directory matching completed, skipping...")
            return
    else:
        print("\nMatching json_dir: ", os.path.basename(json_dir), "N json:", len(json_files))
        
    gdf_dict = {}
    for j, f in enumerate(json_files):
        
        name_root = f.split('.')[0]
        json_path = os.path.join(json_dir, f)
        output_path = os.path.join(out_dir, f)
        
        if verbose and ((j % 1) == 0):
            print("  ", j, "/", len(json_files), "for", os.path.basename(json_dir), "=", name_root)

        # gdf
        gdf_now = gpd.read_file(json_path)
        # drop value if it exists
        gdf_now = gdf_now.drop(columns=['value'])
        # get area
        gdf_now['area'] = gdf_now['geometry'].area
        # initialize iou, id
        gdf_now[iou_field] = -1
        gdf_now[id_field] = -1
        # sort by reverse area
        gdf_now.sort_values(by=['area'], ascending=False, inplace=True)
        gdf_now = gdf_now.reset_index(drop=True)
        # reorder columns (if needed)
        gdf_now = gdf_now[gdf_master_columns]    
        id_set = set([])
                           
        if verbose:
            print("\n")
            print("", j, "file_name:", f)
            print("  ", "gdf_now.columns:", gdf_now.columns)
        
        if j == 0:
            # Establish initial footprints at Epoch0
            # set id
            gdf_now[id_field] = gdf_now.index.values
            gdf_now[iou_field] = 0
            n_new = len(gdf_now)
            n_matched = 0
            id_set = set(gdf_now[id_field].values)
            gdf_master_Out = gdf_now.copy(deep=True)
            # gdf_dict[f] = gdf_now
        else:
            # match buildings in epochT to epochT-1
            # see: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/base.py
            # print("gdf_master;", gdf_dict['master']) #gdf_master)
            gdf_master_Out = gdf_dict['master'].copy(deep=True)
            gdf_master_Edit = gdf_dict['master'].copy(deep=True)
            
            if verbose:
                print("   len gdf_now:", len(gdf_now), "len(gdf_master):", len(gdf_master_Out),
                      "max master id:", np.max(gdf_master_Out[id_field]))
                print("   gdf_master_Edit.columns:", gdf_master_Edit.columns)
        
            new_id = np.max(gdf_master_Edit[id_field]) + 1
            # if verbose:
            #    print("new_id:", new_id)
            idx = 0
            n_new = 0
            n_matched = 0
            for pred_idx, pred_row in gdf_now.iterrows():
                if verbose:
                    if (idx % 1000) == 0:
                        print("    ", name_root, idx, "/", len(gdf_now))
                if super_verbose:
                    # print("    ", i, j, idx, "/", len(gdf_now))
                    print("    ", idx, "/", len(gdf_now))
                idx += 1
                pred_poly = pred_row.geometry
                # if super_verbose:
                #     print("     pred_poly.exterior.coords:", list(pred_poly.exterior.coords))
                    
                # get iou overlap
                iou_GDF = calculate_iou(pred_poly, gdf_master_Edit)
                # iou_GDF = iou.calculate_iou(pred_poly, gdf_master_Edit)
                # print("iou_GDF:", iou_GDF)
                     
                # Get max iou
                if not iou_GDF.empty:
                    max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
                    # sometimes we are get an erroneous id of 0, caused by nan area,
                    #   so check for this
                    max_area = max_iou_row.geometry.area
                    if max_area == 0 or math.isnan(max_area):
                        # print("nan area!", max_iou_row, "returning...")
                        raise Exception("\n Nan area!:", max_iou_row, "returning...")
                        return
                    
                    id_match = max_iou_row[id_field]
                    if id_match in id_set:
                        print("Already seen id! returning...")
                        raise Exception("\n Already seen id!", id_match, "returning...")
                        return
                      
                    # print("iou_GDF:", iou_GDF)
                    if max_iou_row['iou_score'] >= min_iou:
                        if super_verbose:
                            print("    pred_idx:", pred_idx, "match_id:", max_iou_row[id_field],
                                  "max iou:", max_iou_row['iou_score'])
                        # we have a successful match, so set iou, and id
                        gdf_now.loc[pred_row.name, iou_field] = max_iou_row['iou_score']
                        gdf_now.loc[pred_row.name, id_field] = id_match
                        # drop  matched polygon in ground truth
                        gdf_master_Edit = gdf_master_Edit.drop(max_iou_row.name, axis=0)
                        n_matched += 1
                        # # update gdf_master geometry?
                        # # Actually let's leave the geometry the same so it doesn't move around...
                        # gdf_master_Out.at[max_iou_row['gt_idx'], 'geometry'] = pred_poly
                        # gdf_master_Out.at[max_iou_row['gt_idx'], 'area'] = pred_poly.area
                        # gdf_master_Out.at[max_iou_row['gt_idx'], iou_field] = max_iou_row['iou_score']
                      
                    else:
                        # no match, 
                        if super_verbose:
                            print("    Minimal match! - pred_idx:", pred_idx, "match_id:",
                                  max_iou_row[id_field], "max iou:", max_iou_row['iou_score'])
                            print("      Using new id:", new_id)
                        if (new_id in id_set) or (new_id == 0):
                            raise Exception("trying to add an id that already exists, returning!")
                            return
                        gdf_now.loc[pred_row.name, iou_field] = 0
                        gdf_now.loc[pred_row.name, id_field] = new_id
                        id_set.add(new_id)
                        # update master, cols = [id_field, iou_field, 'area', 'geometry']
                        gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly]
                        new_id += 1
                        n_new += 1
                       
                else:
                    # no match (same exact code as right above)
                    if super_verbose:
                        print("    pred_idx:", pred_idx, "no overlap, new_id:", new_id)
                    if (new_id in id_set) or (new_id == 0):
                        raise Exception("trying to add an id that already exists, returning!")
                        return
                    gdf_now.loc[pred_row.name, iou_field] = 0
                    gdf_now.loc[pred_row.name, id_field] = new_id
                    id_set.add(new_id)
                    # update master, cols = [id_field, iou_field, 'area', 'geometry']
                    gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly]
                    new_id += 1
                    n_new += 1
                    
        # print("gdf_now:", gdf_now)
        gdf_dict[f] = gdf_now
        gdf_dict['master'] = gdf_master_Out

        # save!
        if len(gdf_now) > 0:
            gdf_now.to_file(output_path, driver="GeoJSON")
        else:
            print("Empty dataframe, writing empty gdf", output_path)
            open(output_path, 'a').close()

        if verbose:
            print("  ", "N_new, N_matched:", n_new, n_matched)
         
    return 
  

def sn7_convert_geojsons_to_csv(json_dirs, output_csv_path, population='proposal'):
    '''
    Convert jsons to csv
    Population is either "ground" or "proposal" 
    '''
    
    first_file = True  # switch that will be turned off once we process the first file
    for json_dir in tqdm.tqdm(json_dirs):
        json_files = sorted(glob.glob(os.path.join(json_dir, '*.geojson')))
        for json_file in tqdm.tqdm(json_files):
            try:
                df = gpd.read_file(json_file)
            except (fiona.errors.DriverError):
                message = '! Invalid dataframe for %s' % json_file
                print(message)
                continue
                #raise Exception(message)
            if population == 'ground':
                file_name_col = df.image_fname.apply(lambda x: os.path.splitext(x)[0])
            elif population == 'proposal':
                file_name_col = os.path.splitext(os.path.basename(json_file))[0]
            else:
                raise Exception('! Invalid population')
            df = gpd.GeoDataFrame({
                'filename': file_name_col,
                'id': df.Id.astype(int),
                'geometry': df.geometry,
            })
            if len(df) == 0:
                message = '! Empty dataframe for %s' % json_file
                print(message)
                #raise Exception(message)

            if first_file:
                net_df = df
                first_file = False
            else:
                net_df = net_df.append(df)
                
    net_df.to_csv(output_csv_path, index=False)
    return net_df