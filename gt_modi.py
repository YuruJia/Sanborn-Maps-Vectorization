from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import numpy as np
import matplotlib.pyplot as plt

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

# Disentangle the individual channels
band1 = original_raster.GetRasterBand(1) # Red channel
band2 = original_raster.GetRasterBand(2) # Green channel
band3 = original_raster.GetRasterBand(3) # Blue channel

#Tranfsformation into numpy arrays
b1 = band1.ReadAsArray(); b1 = b1.astype(np.uint8)
b2 = band2.ReadAsArray(); b2 = b2.astype(np.uint8)
b3 = band3.ReadAsArray(); b3 = b3.astype(np.uint8)

image = np.stack((b1,b2,b3),axis=2)

plt.imshow(image)
plt.show()








