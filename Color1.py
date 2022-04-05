from osgeo import gdal
from osgeo import osr
from osgeo import ogr
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
b1 = band1.ReadAsArray()
b2 = band2.ReadAsArray()
b3 = band3.ReadAsArray()

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

#Create Color Space

"""
fig = plt.figure()
axis = fig.add_subplot(1,1,1, projection ="3d")

pixel_colors = raster_data.reshape((np.shape(raster_data)[1]*np.shape(raster_data)[2], 4))
norm = colors.Normalize(vmin =-1., vmax = 1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(b1_flat, b2_flat, b3_flat, facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()
"""


#plt.plot(bin_edges[0:-1], histogram)
#plt.show()






#Writing output file
#drive = gdal.GetDriverByName("GTiff")
#output_path = 'C:/Users/Matthias Walder/Desktop/Geologie/Multimedia Cartography/Project/Graphical6751_test.tif'
#outdata = driver.Create(output_path, raster_map[2], raster_map.shape[1], 3,gdal.GDT_Byte,['COMPRESS=LZW'])
#outdata.SetGeoTransform(transform)
#outdata.SetProjection(projection)
#outdata.GetRasterBand(1).WriteArray(raster_map[0])
#outdata.FlushCache()
#outdata = None
