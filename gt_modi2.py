from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Load Files
original_path = 'C:/Users/Matthias Walder/Desktop/Geologie/Multimedia Cartography/Project/Graphical/GT_modification/original.tif'
gt_path = 'C:/Users/Matthias Walder/Desktop/Geologie/Multimedia Cartography/Project/Graphical/GT_modification/67511_256_1536.tif'

original_map = gdal.Open(original_path)
gt_map = gdal.Open(gt_path)
original_raster = original_map.ReadAsArray()
gt_raster = gt_map.ReadAsArray()

original_projection = original_map.GetProjection()
original_transform = original_map.GetGeoTransform()
gt_projection = gt_map.GetProjection()
gt_transform = gt_map.GetGeoTransform()

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

#extract statistical information for connected parts (all the building parts)
num_labels, labels, stats, centroids  = cv2.connectedComponentsWithStats(gt_raster, 4, cv2.CV_32S)

#iterate through all components and define for every component whether its a brick
thresh_brick_min = [164, 145, 2135]
thresh_brick_max = [210, 180, 172]


for label_idx in range(1, num_labels):
    #define the coordinates for the bounding rectangle for the current component
    x = stats[label_idx, cv2.CC_STAT_LEFT]
    y = stats[label_idx, cv2.CC_STAT_TOP]
    w = stats[label_idx, cv2.CC_STAT_WIDTH]
    h = stats[label_idx, cv2.CC_STAT_HEIGHT]
    
    img_crop = image[y:y+w, x:x+h,:]
    plt.title('Original RGB image')
    plt.imshow(img_crop)
    plt.show()
    
    R = stat.mode(img_crop[:,:,0].flatten())
    G = stat.mode(img_crop[:,:,1].flatten());
    B = stat.mode(img_crop[:,:,2].flatten());
    rgb = [R, G, B]
    print(rgb)
    
    #Create Rectangle
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Output", output)
    cv2.waitKey(1500)
    
    label_mask = (labels == label_idx)   
    #set all brick pixels to 255
    if rgb > thresh_brick_min and rgb < thresh_brick_max:
        gt_raster[label_mask[:]] = 255
    else: 
        gt_raster[label_mask[:]] = 0

plt.title('Brick Mask')
plt.imshow(gt_raster)
plt.show()


""" 
#make all brick components white
gt_raster[gt_raster != 20] = 0
gt_raster[gt_raster == 20] = 255   
"""   


