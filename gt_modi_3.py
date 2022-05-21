from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

dir_original = os.fsencode('loop_test/tif')
dir_gt = os.fsencode('loop_test/label')

#Define Functions

def buildings(raster_gt, raster_input, brick_min, brick_max, frame_min, frame_max):  

    #extract statistical information for connected parts (all the building parts)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raster_gt, 4, cv2.CV_32S)

    for label_idx in range(1, num_labels):
            
        #img_crop = image[y:y+w, x:x+h,:]
        label_mask = (labels == label_idx)
        copy = np.copy(raster_input)
        copy[~label_mask] = 0
        plt.title('Original RGB image')
        plt.imshow(copy)
        plt.show()
        print(copy.shape)
        R = copy[:,:,0].flatten(); R = R[R != 0]; R = stat.mode(R)
        G = copy[:,:,1].flatten(); G = G[G != 0]; G = stat.mode(G)
        B = copy[:,:,2].flatten(); B = B[B != 0]; B = stat.mode(B)
        rgb = [R, G, B]
        print(rgb)
             
        #set all brick pixels to 255
        if rgb > brick_min and rgb < brick_max:
            raster_gt[label_mask[:]] = 255
            print("We found a brick")
        elif rgb > frame_min and rgb < frame_max:
            raster_gt[label_mask[:]] = 125
            print(" We found framwork")
        else: 
            raster_gt[label_mask[:]] = 0
            print("That's something else")
            

def raster_output(path, projection, transform, height, width, array):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, width, height, 1, gdal.GDT_Int16, ['COMPRESS=LZW'])

    outdata.SetGeoTransform(transform)
    outdata.SetProjection(projection)

    outdata.GetRasterBand(1).WriteArray(array)

    outdata.FlushCache()
    outdata = None

def create_pathlist(original_folder, gt_folder): 

    pathlist = []
    #look for double filenames and add them to the pathlist
    for file in os.listdir(original_folder):
        original_filename = os.fsdecode(file)
        for file2 in os.listdir(gt_folder):
            gt_filename = os.fsdecode(file2)
            if original_filename == gt_filename:
                pathlist.append(original_filename)
    
    return pathlist


path = create_pathlist(dir_original, dir_gt)
print(path)

#iterate through files
for file in path:
    original_path = "loop_test/tif/" + file
    gt_path = "loop_test/label/" + file
    output_path = "loop_test/modified/" + file

    original_map = gdal.Open(original_path)
    gt_map = gdal.Open(gt_path)
    original_raster = original_map.ReadAsArray()
    gt_raster = gt_map.ReadAsArray()

    original_projection = original_map.GetProjection()
    original_transform = original_map.GetGeoTransform()
    gt_projection = gt_map.GetProjection()
    gt_transform = gt_map.GetGeoTransform()

    #Define threshold values
    thresh_brick_min = [167, 145, 138]
    thresh_brick_max = [192, 172, 162]

    thresh_frame_min = [146, 138, 100]
    thresh_frame_max = [175, 170, 130]

    # Disentangle the individual channels (original)
    band1 = original_map.GetRasterBand(1) # Red channel
    band2 = original_map.GetRasterBand(2) # Green channel
    band3 = original_map.GetRasterBand(3) # Blue channel

    #Tranfsformation into numpy arrays
    b1 = band1.ReadAsArray(); b1 = b1.astype(np.uint8)
    b2 = band2.ReadAsArray(); b2 = b2.astype(np.uint8)
    b3 = band3.ReadAsArray(); b3 = b3.astype(np.uint8)

    #restacking the bands to display and further processing
    image = np.stack((b1,b2,b3),axis=2)
    plt.imshow(image)
    plt.show()


    # call functions

    buildings(gt_raster, image, thresh_brick_min, thresh_brick_max, thresh_frame_min, thresh_frame_max)
    raster_output(output_path, original_projection, original_transform, gt_raster.shape[0], gt_raster.shape[1], gt_raster)

    plt.title('Mask')
    plt.imshow(gt_raster)
    plt.show()