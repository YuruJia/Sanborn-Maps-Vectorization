from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import statistics as stat
import fiona, copy
import os
from numpy import genfromtxt

import numpy as np
import csv

#Get the filename in order to save generated files with the same number

raster_folder = os.fsencode('data/tif')
vector_folder = os.fsencode('data/label')


def get_filenames(original_folder, gt_folder): 

    pathlist = []
    #look for double filenames and add them to the pathlist
    for file in os.listdir(original_folder):
        original_filename = os.fsdecode(file[:4])
        for file2 in os.listdir(gt_folder):
            gt_filename = os.fsdecode(file2[:4])
            if original_filename == gt_filename:
                pathlist.append(original_filename)
                break
    
    return pathlist

brick_file = genfromtxt('data/brick_thresh.csv', delimiter=',') #load the threshold values from the csv files
frame_file = genfromtxt('data/frame_thresh.csv', delimiter=',')

loop = get_filenames(raster_folder, vector_folder)  # extract the number of each sheet and store it in loop variable

for sheets in loop: #iterate over every mapsheet

    #get files and load projections etc.
    
    vector_path = 'data/label/' + str(sheets) +'.shp'
    raster_path = 'data/tif/' + str(sheets)+ '.tif'
    
    vector_map = ogr.Open(vector_path)
    layer = vector_map.GetLayer()
    
    raster_map = gdal.Open(raster_path)
    geot = raster_map.GetGeoTransform()
    geoproj = raster_map.GetProjection()
    
    raster_arr = raster_map.ReadAsArray()
    
    print(sheets) #current sheet
    print(frame_file)
    pos = 0; pos2 = 0
    brick_label = (brick_file[:,0]) #extract row number of current sheet
    frame_label = (frame_file[:,0])
    for i in range(len(brick_label)):
        if abs(brick_label[i]) == int(sheets):
            pos = i
            print(pos)
        if abs(frame_label[i]) == int(sheets):
            pos2 = i
            print(pos2)
                    
    brick_min = [brick_file[pos,1],brick_file[pos,2],brick_file[pos,3]] #extract the threshold range values for brick and frame
    brick_max = [brick_file[pos,4],brick_file[pos,5],brick_file[pos,6]]
    frame_min = [frame_file[pos2,1],frame_file[pos2,2],frame_file[pos2,3]]
    frame_max = [frame_file[pos2,4],frame_file[pos2,5],frame_file[pos2,6]]
    
    sheet_stats = {} 
    
    for i in range(layer.GetFeatureCount()):
    # apply a filter to the layer which, in this case based on the feature id, i.e., in each iteration the layer will only be comprised of a single feature that can be rasterized
        layer.SetAttributeFilter("FID = " + str(i)) 
    
    # we use a raster that resides in the computer memory, no need to write it out to disk here
        in_memory_path = '/vsimem/rasterized.tif'
        drv_mem = gdal.GetDriverByName("MEM")
        chn_ras_ds = drv_mem.Create(in_memory_path, raster_map.RasterXSize, raster_map.RasterYSize, 1, gdal.GDT_Byte)
    
        chn_ras_ds.SetGeoTransform(geot)
        chn_ras_ds.SetProjection(geoproj)

        gdal.RasterizeLayer(chn_ras_ds, [1], layer)
    
        band = chn_ras_ds.GetRasterBand(1)
        band_arr = band.ReadAsArray()
        print(np.sum(band_arr))
    
        building_mask = band_arr == 255
        print(np.sum(building_mask)) # how many pixels are covered by this building
        
        raster_arr_red = raster_arr[0,:,:]
        raster_arr_green = raster_arr[1,:,:]
        raster_arr_blue = raster_arr[2,:,:]
        
        red_masked = raster_arr_red[building_mask]
        green_masked = raster_arr_green[building_mask]
        blue_masked = raster_arr_blue[building_mask]

        R = stat.mode(red_masked)
        G = stat.mode(green_masked)
        B = stat.mode(blue_masked)
        
        #choose building_type based on thresholding range: 1 = brick, 2 = framework
        building_type = 0
        if [R, G, B] >= brick_min and [R, G, B] <= brick_max:
            building_type = 1
        elif [R, G, B] >= frame_min and [R, G, B] <= frame_max:
            building_type = 2
        
        print(R, G, B, building_type)
        sheet_stats[i] = (R, G, B, building_type)

    with open(str(sheets) +'.csv', "w", newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["id", "red", "green", "blue","building"])
        for key, value in sheet_stats.items():
            spamwriter.writerow([key, value[0], value[1], value[2], value[3]])

