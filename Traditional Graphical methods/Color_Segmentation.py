from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from skimage import io
import skimage.io
import skimage.color
import skimage.filters
import cv2
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# Load file
input_path = 'data/6751.tif'
mapfile = gdal.Open(input_path)
raster_data = mapfile.ReadAsArray()
projection = mapfile.GetProjection()
transform = mapfile.GetGeoTransform()

# Disentangle the individual channels
band1 = mapfile.GetRasterBand(1) # Red channel
band2 = mapfile.GetRasterBand(2) # Green channel
band3 = mapfile.GetRasterBand(3) # Blue channel

#Tranfsformation into numpy arrays
b1 = band1.ReadAsArray(); b1 = b1.astype(np.uint8)
b2 = band2.ReadAsArray(); b2 = b2.astype(np.uint8)
b3 = band3.ReadAsArray(); b3 = b3.astype(np.uint8)

#Collapse Array into a 1D array and remove zero values (dark border of image)
b1_flat = b1.flatten(); b1_flat_hist = b1_flat[b1_flat != 0]
b2_flat = b2.flatten(); b2_flat_hist = b2_flat[b2_flat != 0]
b3_flat = b3.flatten(); b3_flat_hist = b3_flat[b3_flat != 0]

#Create Histograms
fig, axs = plt.subplots(3, figsize = (10,7), sharey = True, sharex = True)
axs[0].hist(b1_flat_hist, bins = 256)
axs[0].set_title('Red Band Histogram')
axs[1].hist(b2_flat_hist, bins = 256)
axs[1].set_title('Green Band Histogram')
axs[2].hist(b3_flat_hist, bins = 256)
axs[2].set_title('Blue Band Histogram')

axs[2].set(xlabel ='Grayscale Value (Brightness') 

for ax in axs.flat:
    ax.set(ylabel='Pixel Counts')

plt.show() 


#Crop file for color space

image = np.stack((b1,b2,b3),axis=2)

a, b, s = 1500, 1500, 800# offset, offset, size for cropping
img_crop = image[a:a+s, b:b+s,:]
io.imsave("crop_sanborn.jpg",img_crop)
plt.figure(figsize=(8,8))
plt.title('Original RGB image')
plt.imshow(img_crop)
plt.show()

# Perform Image Segmentation based on RGB-threshold values - Figuring out The correct range of values! 
# Result 12 for brick, 18 for framework


# Remove the small parts
fig = plt.figure()
tolerance = 18
thresh_brick = [179,159,150] # Set Threshold
#thresh_frame = [153,146,104] # Set Threshold
thresh_brick_min = np.subtract(thresh_brick, tolerance); thresh_brick_max = np.add(thresh_brick, tolerance) # Widen tolerance for thresholding
#thresh_frame_min = np.subtract(thresh_frame, tolerance); thresh_frame_max = np.add(thresh_frame, tolerance)

mask_brick = cv2.inRange(img_crop, thresh_brick_min, thresh_brick_max)
#mask_frame = cv2.inRange(img_crop, thresh_frame_min, thresh_frame_max)    
output = cv2.bitwise_and(img_crop, img_crop, mask = mask_brick)
#output = cv2.bitwise_and(img_crop, img_crop, mask = mask_frame)


plt.imshow(output)
plt.show()


# Create a Mask (1D-Array) with binary value range
red_band = output[:,:,0]; green_band = output[:,:,1]; blue_band = output[:,:,2]


for cell in np.nditer(red_band, op_flags=['readwrite']):
    if cell[...] < 15:
        cell[...] = 0
    else:
        cell[...] = 1

for cell in np.nditer(green_band, op_flags=['readwrite']):
    if cell[...] < 15:
        cell[...] = 0
    else:
        cell[...] = 1
        
for cell in np.nditer(blue_band, op_flags=['readwrite']):
    if cell[...] < 15:
        cell[...] = 0
    else:
        cell[...] = 1

mask = green_band*blue_band*red_band
mask = mask*255

# Modify the mask


area_threshold = 7

num_labels, labels, stats, _  = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S) #num_labels = total number of unique labels / labels has the same spatial dimension as raster_data / for each location in labels


for label_idx in range(num_labels): 
    label_mask = (labels == label_idx)
    
    if stats[label_idx, 4] <= area_threshold:
         mask[label_mask[:]] = 0

plt.imshow(mask, cmap = "gray")
plt.show()

kernel = np.ones((3,3,), np.uint8)
mask = cv2.dilate(mask, kernel, iterations= 5)
mask = cv2.erode(mask, kernel, iterations = 5)

plt.imshow(mask, cmap = "gray")
plt.show()

#write to raster_file

def write_raster_simple(path, projection, transform, height, width, array):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, width, height, 1, gdal.GDT_Int16, ['COMPRESS=LZW'])

    outdata.SetGeoTransform(transform)
    outdata.SetProjection(projection)

    outdata.GetRasterBand(1).WriteArray(array)

    outdata.FlushCache()
    outdata = None

write_raster_simple("export.tif", projection, transform, mask.shape[0], mask.shape[1], mask)


