from osgeo import gdal, ogr, osr
import numpy as np
import cv2
import fiona
import matplotlib.pyplot as plt

#This Script adds the geospatial information to the Neural Network output

path_brick = 'data/brick.tif'
path_frame = 'data/frame.tif'
path_edges = 'data/edges.tif'
path_meta = 'data/6909.tif'

def write_raster(path, projection, transform, height, width, array):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, width, height, 1, gdal.GDT_Int16, ['COMPRESS=LZW'])

    outdata.SetGeoTransform(transform)
    outdata.SetProjection(projection)

    outdata.GetRasterBand(1).WriteArray(array)

    outdata.FlushCache()
    outdata = None

brick = gdal.Open(path_brick); frame = gdal.Open(path_frame); edge = gdal.Open(path_edges); meta = gdal.Open(path_meta)
geot = meta.GetGeoTransform(); geoproj = meta.GetProjection()
band_brick = brick.GetRasterBand(1); band_frame = frame.GetRasterBand(1); band_edge = edge.GetRasterBand(1)
brick_arr = band_brick.ReadAsArray(); frame_arr = band_frame.ReadAsArray(); edge_arr = edge.ReadAsArray()

#Saving the new georeferenced brick_file
write_raster("data/brick_ref.tif", geoproj, geot, brick_arr.shape[0], brick_arr.shape[1], brick_arr)
write_raster("data/edges_ref.tif", geoproj, geot, edge_arr.shape[0], edge_arr.shape[1], edge_arr)


