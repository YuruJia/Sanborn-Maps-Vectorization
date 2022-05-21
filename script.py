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
    
    return filename

brick_file = genfromtxt('brick_thresh.csv', delimiter=',')
frame_file = genfromtxt('frame_thresh.csv', delimiter=',')

loop = get_filenames(raster_folder, gt_folder)

for sheets in loop:
    print("hey")

    vector_path = 'data/label/' + str(sheets) +'.shp'
    raster_path = 'data/tif/' + str(sheets)+ '.tif'


    vector_map = ogr.Open(vector_path)
    raster_map = gdal.Open(raster_path)

    layer = vector_map.GetLayer()
    geot = raster_map.GetGeoTransform()
    geoproj = raster_map.GetProjection()

    raster_arr = raster_map.ReadAsArray()

    brick_min = brick_file

    color_values = {}


for i in range(layer.GetFeatureCount()):
    # apply a filter to the layer which, in this case based on the feature id, i.e., in each iteration the layer will only be comprised of a single feature that can be rasterized
    layer.SetAttributeFilter("FID = " + str(i)) 
    
    # we use a raster that resides in the computer memory, no need to write it out to disk here
    in_memory_path = '/vsimem/rasterized.tif'
    drv_mem = gdal.GetDriverByName("MEM")
    chn_ras_ds = drv_mem.Create(in_memory_path, raster_map.RasterXSize, raster_map.RasterYSize, 1, gdal.GDT_Byte)
    
    # if you like, you can also use this piece of code, which writes out the single rasters, one for each building
    # make sure that the path makes sense and to comment the code above
    
    """
    in_memory_path = 'rasterized/rasterized_' + str(i) + '.tif'
    drv_mem = gdal.GetDriverByName("GTiff")
    chn_ras_ds = drv_mem.Create(in_memory_path, raster_map.RasterXSize, raster_map.RasterYSize, 1, gdal.GDT_Byte)
    """
    
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
    
    """
    avg_red = np.average(red_masked)
    avg_green = np.average(green_masked)
    avg_blue = np.average(blue_masked)
    """
    
    R = stat.mode(red_masked)
    G = stat.mode(green_masked)
    B = stat.mode(blue_masked)
    
    print(R, G, B)
    color_values[i] = (R, G, B)
    


    
with open(str(filename[:4]) +'.csv', "w", newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(["id", "red", "green", "blue"])
    for key, value in color_values.items():
        spamwriter.writerow([key, value[0], value[1], value[2]])
 
input_file = csv.reader(open(str(filename[:4]) + '.csv', 'r'))
r = []; g = []; b = []
next(input_file) #skip header
for row in input_file: #create three lists, one for each color
    r.append(row[1]); g.append(row[2]); b.append(row[3])
rgb = np.zeros((len(r),3))
for i in range(len(r)):
    rgb[i][0] = r[i]
    rgb[i][1] = g[i]
    rgb[i][2] = b[i]
    
 

    
    

    
    
    
    
    
    
    
    
    
    
    