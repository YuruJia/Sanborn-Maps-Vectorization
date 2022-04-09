from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from skimage import io
import cv2
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# Load file
input_path = 'C:/Users/Matthias Walder/Desktop/Geologie/Multimedia Cartography/Project/material_sanborn/sheets/6751.tif'
mapfile = gdal.Open(input_path)
raster_data = mapfile.ReadAsArray()

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

fig, axes = plt.subplots(nrows=2, ncols=4)
for tolerance in range(4,20,2):
    
    thresh_brick = [179,159,150] # Set Threshold
    #thresh_frame = [153,146,104] # Set Threshold
    thresh_brick_min = np.subtract(thresh_brick, tolerance); thresh_brick_max = np.add(thresh_brick, tolerance) # Widen tolerance for thresholding
    #thresh_frame_min = np.subtract(thresh_frame, tolerance); thresh_frame_max = np.add(thresh_frame, tolerance)

    mask_brick = cv2.inRange(img_crop, thresh_brick_min, thresh_brick_max)
    #mask_frame = cv2.inRange(img_crop, thresh_frame_min, thresh_brick_max)
    
    output = cv2.bitwise_and(img_crop, img_crop, mask = mask_brick)
    
    plot_number = int(240 + ((tolerance-2)/2))
    ax = fig.add_subplot(plot_number)
    ax.title.set_text(tolerance)
    plt.imshow(output)

plt.show() 
 
fig2, axes2 = plt.subplots(nrows=2, ncols=4)

for tolerance in range(10,26,2):
    
    #thresh_brick = [179,159,150] # Set Threshold
    thresh_frame = [153,146,104] # Set Threshold
    #thresh_brick_min = np.subtract(thresh_brick, tolerance); thresh_brick_max = np.add(thresh_brick, tolerance) # Widen tolerance for thresholding
    thresh_frame_min = np.subtract(thresh_frame, tolerance); thresh_frame_max = np.add(thresh_frame, tolerance)

    #mask_brick = cv2.inRange(img_crop, thresh_brick_min, thresh_brick_max)
    mask_frame = cv2.inRange(img_crop, thresh_frame_min, thresh_frame_max)
    
    output2 = cv2.bitwise_and(img_crop, img_crop, mask = mask_frame)
    
    plot_number = int(240 + ((tolerance-8)/2))
    ax = fig2.add_subplot(plot_number)
    ax.title.set_text(tolerance)
    plt.imshow(output2) 
     

plt.show()

#Create Color Space

"""
fig = plt.figure()
axis = fig.add_subplot(1,1,1, projection ="3d")

pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
norm = colors.Normalize(vmin =-1., vmax = 1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(image[:,:,0].flatten()/255, image[:,:,1].flatten()/255, image[:,:,2].flatten()/255, facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()



#plt.plot(bin_edges[0:-1], histogram)
#plt.show()
"""





#Writing output file
#drive = gdal.GetDriverByName("GTiff")
#output_path = 'C:/Users/Matthias Walder/Desktop/Geologie/Multimedia Cartography/Project/Graphical6751_test.tif'
#outdata = driver.Create(output_path, raster_map[2], raster_map.shape[1], 3,gdal.GDT_Byte,['COMPRESS=LZW'])
#outdata.SetGeoTransform(transform)
#outdata.SetProjection(projection)
#outdata.GetRasterBand(1).WriteArray(raster_map[0])
#outdata.FlushCache()
#outdata = None
