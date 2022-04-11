from osgeo import gdal, ogr, osr
from skimage import io
import cv2
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import csv

# Load file
input_path = 'C:/Users/Matthias Walder/Desktop/Geologie/Multimedia Cartography/Project/material_sanborn/sheets/6751.tif'
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

image = np.stack((b1,b2,b3),axis=2)

a, b, s = 1500, 1500, 800# offset, offset, size for cropping
img_crop = image[a:a+s, b:b+s,:]
io.imsave("crop_sanborn.jpg",img_crop)
plt.figure(figsize=(8,8))
plt.title('Original RGB image')
plt.imshow(img_crop)
plt.show()

color_explore = np.zeros((150,150,3), np.uint8)  
color_selected = np.zeros((150,150,3), np.uint8)
thresh_frame = [0,0,0]

#save selected color RGB in file
def write_to_file(R,G,B):
	f = open("saved_color.txt", "a")
	RGB_color=str(R) + "," + str(G) + "," + str(B) + str("\n")
	f.write(RGB_color); global color ; color = [R,G,B]
	f.close()


#Mouse Callback function
def show_color(event,x,y,flags,param): 
	
	B=img[y,x][0]
	G=img[y,x][1]
	R=img[y,x][2]
	color_explore [:] = (B,G,R)

	if event == cv2.EVENT_LBUTTONDOWN:
		color_selected [:] = (B,G,R)


	if event == cv2.EVENT_RBUTTONDOWN:
		B=color_selected[10,10][0]
		G=color_selected[10,10][1]
		R=color_selected[10,10][2]
		print(R,G,B)
		write_to_file(R,G,B)
		print(hex(R),hex(G),hex(B))
	
#live update color with cursor
cv2.namedWindow('color_explore')
cv2.resizeWindow("color_explore", 50,50);

#Show selected color when left mouse button pressed
cv2.namedWindow('color_selected')
cv2.resizeWindow("color_selected", 50,50);

#image window for sample image
cv2.namedWindow('image')

#sample image path
img_path="crop_sanborn.jpg"

#read sample image
img=cv2.imread(img_path)

#mouse call back function declaration
cv2.setMouseCallback('image',show_color)

#while loop to live update
while (1):
	
	cv2.imshow('image',img)
	cv2.imshow('color_explore',color_explore)
	cv2.imshow('color_selected',color_selected)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()

fig, axes = plt.subplots(nrows=2, ncols=4)
for tolerance in range(4,20,2):
    
    thresh_brick = color # Set Threshold
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

write_raster_simple("data/export.tif", projection, transform, mask.shape[0], mask.shape[1], mask)

#write to shapefile

srs = osr.SpatialReference()
srs.ImportFromWkt(projection)

polygons_path = "data/polygons.shp"
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource( polygons_path )
dst_layer = dst_ds.CreateLayer("polygons", srs = srs )
gdal.Polygonize(mask, mask, dst_layer, -1, [], callback=None)

dst_layer = None
dst_ds = None