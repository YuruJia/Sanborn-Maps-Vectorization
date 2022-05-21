from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import statistics as stat
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt

import numpy as np
import csv

#Get the filename in order to save generated files with the same number

raster_folder = os.fsencode('data/tif')
vector_folder = os.fsencode('data/modified/label_2')


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
    
loop = get_filenames(raster_folder, vector_folder)  # extract the number of each sheet and store it in loop variable

for sheets in loop: #iterate over every mapsheet

    vector_path = 'data/modified/label_2/' + str(sheets) +'.shp'
    raster_path = 'data/tif/' + str(sheets)+ '.tif'
    
    vector_map = ogr.Open(vector_path)
    layer = vector_map.GetLayer()
    
    raster_map = gdal.Open(raster_path)
    geot = raster_map.GetGeoTransform()
    geoproj = raster_map.GetProjection()
    
    raster_arr = raster_map.ReadAsArray()
    
    in_memory_path = 'data/modified/tif/' + sheets + '.tif'
    drv_mem = gdal.GetDriverByName("GTiff")
    chn_ras_ds = drv_mem.Create(in_memory_path, raster_map.RasterXSize, raster_map.RasterYSize, 1, gdal.GDT_Byte)
    
    chn_ras_ds.SetGeoTransform(geot)
    chn_ras_ds.SetProjection(geoproj)

    gdal.RasterizeLayer(chn_ras_ds, [1], layer, options=['ATTRIBUTE=type_build'])
    band = chn_ras_ds.GetRasterBand(1)
         
    brick_array = band.ReadAsArray()
    frame_array = brick_array.copy()
        
    for cell in np.nditer(brick_array, op_flags=['readwrite']):
        if cell[...] == 1:
            cell[...] = 255
        else:
            cell[...] = 0
            
    for cell in np.nditer(frame_array, op_flags=['readwrite']):
        if cell[...] == 2:
            cell[...] = 255
        else:
            cell[...] = 0
    
    brick_out_path = 'data/modified/tif/'+str(sheets)+'_brick.tif'
    frame_out_path = 'data/modified/tif/'+str(sheets)+'_frame.tif'
    
    brick_channel = drv_mem.Create(brick_out_path, raster_map.RasterXSize, raster_map.RasterYSize, 1, gdal.GDT_Byte)
    brick_channel.SetGeoTransform(geot)
    brick_channel.SetProjection(geoproj)
    brick_out = brick_channel.GetRasterBand(1)
    brick_out.WriteArray(brick_array)
    outdata = None
    
    frame_channel = drv_mem.Create(frame_out_path, raster_map.RasterXSize, raster_map.RasterYSize, 1, gdal.GDT_Byte)
    frame_channel.SetGeoTransform(geot)
    frame_channel.SetProjection(geoproj)
    frame_out = frame_channel.GetRasterBand(1)
    frame_out.WriteArray(frame_array)
    outdata = None    
       
    
    