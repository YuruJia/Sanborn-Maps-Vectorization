from osgeo import gdal, ogr, osr
import numpy as np
import cv2
import fiona
import matplotlib.pyplot as plt


path_meta = 'data/6909.tif'; meta = gdal.Open(path_meta); geot = meta.GetGeoTransform(); geoproj = meta.GetProjection() 

def write_raster_simple(path, projection, transform, height, width, array):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, width, height, 1, gdal.GDT_Int16, ['COMPRESS=LZW'])

    outdata.SetGeoTransform(transform)
    outdata.SetProjection(projection)

    outdata.GetRasterBand(1).WriteArray(array)

    outdata.FlushCache()
    outdata = None


def noise_reduction(path_in):
    
    dataset = gdal.Open(path_in)
    band = dataset.GetRasterBand(1)
    band_array = band.ReadAsArray()
    band_array = np.uint8(band_array)
    area_threshold = 4

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(band_array, 4, cv2.CV_32S)

    for label_idx in range(num_labels): 
        label_mask = (labels == label_idx)
            
        if stats[label_idx, 4] <= area_threshold:
            band_array[label_mask[:]] = 0

    plt.imshow(band_array, cmap = "gray")
    plt.show()
    
    kernel = np.ones((2,2,), np.uint8)
    band_array = cv2.dilate(band_array, kernel, iterations= 3)
    #band_array = cv2.erode(band_array, kernel, iterations = 5)

    plt.imshow(band_array, cmap = "gray")
    plt.show()  
    write_raster_simple('edge_thick.tif', geoproj, geot, band_array.shape[0], band_array.shape[1], band_array)

#noise_reduction('data/edges.tif')

def vectorize(path_in, path_out):

    print("Loading files")
    dataset = gdal.Open(path_in)
    band = dataset.GetRasterBand(1)

    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)

    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( path_out )
    dst_layer = dst_ds.CreateLayer("Polygons", srs = srs )
    gdal.Polygonize(band, band, dst_layer, -1, [], callback=None)
    print("Finishing Vectorization")
    dst_layer = None
    dst_ds = None

vectorize('data/edges.tif','data/edges_lines.shp')    


   