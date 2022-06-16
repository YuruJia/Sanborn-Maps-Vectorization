# -*-coding:utf-8-*-

import sys
import os
from osgeo import gdal
from PIL import Image
import numpy as np
# from gdalconst import *
from osgeo import gdalconst

from time import time

sys.path.append('..')
import shutil

import threading
import multiprocessing


class CSaveSplitJpg(object):
    def __init__(self, **kargs):
        self.args = dict(**kargs)

    def start_save(self):
        mem_driver = gdal.GetDriverByName("MEM")
        if mem_driver is None:
            return

        mem_outdata = mem_driver.Create("", self.args["sizeX"],
                                        self.args["sizeY"],
                                        self.args["rasterCount"],
                                        self.args["datatype"])

        bandArr = []
        count = 1
        for k in range(0, self.args["rasterCount"]):
            bandArr.append(count)
            count += 1

        if self.args["startY"] + self.args["sizeY"] > self.args["height"] and \
                self.args["startX"] + self.args["sizeX"] > self.args["width"]:
            mem_outdata.WriteRaster(0, 0, self.args["width"] - self.args["startX"],
                                    self.args["height"] - self.args["startY"], self.args["datas"],
                                    None, None, self.args["datatype"],
                                    bandArr)

        elif self.args["startY"] + self.args["sizeY"] > self.args["height"]:
            mem_outdata.WriteRaster(0, 0, self.args["sizeX"], self.args["height"] - self.args["startY"],
                                    self.args["datas"], None, None, self.args["datatype"], bandArr)

        elif self.args["startX"] + self.args["sizeX"] > self.args["width"]:
            mem_outdata.WriteRaster(0, 0, self.args["width"] - self.args["startX"],
                                    self.args["sizeY"], self.args["datas"], None, None, self.args["datatype"], bandArr)

        else:
            mem_outdata.WriteRaster(0, 0, self.args["sizeX"], self.args["sizeY"],
                                    self.args["datas"], None, None, self.args["datatype"], bandArr)

        dr_jpg = gdal.GetDriverByName('GTiff')
        dr_jpg.CreateCopy(self.args["outputFile"], mem_outdata)
        del mem_outdata
        del dr_jpg


def run_save_split_jpg(param):
    while True:
        exce_cli = param.get()
        if not isinstance(exce_cli, CSaveSplitJpg):
            if exce_cli == "end":
                break
        else:
            exce_cli.start_save()


class PictCut(object):

    def imageCutByRasterIO(self, inputFile, outputPath, bufferX, bufferY, sizeX, sizeY, count,sheets):
        print("start cutting image...")
        name = os.path.basename(inputFile).split(".")[0]
        inDataset = gdal.Open(inputFile, gdal.GA_ReadOnly)

        if inDataset is None:
            #            g_log.fatal('cannot open ' + inputFile)
            return 1

        rasterCount = inDataset.RasterCount

        if sheets:
            band_list = [1, 2, 3]
        else:
            band_list = [1]
        # band_list = [1, 2, 3]
        # band_list = [1]
        geo_info = inDataset.GetGeoTransform()
        width = inDataset.RasterXSize
        height = inDataset.RasterYSize
        grid_x = int(width / (sizeX - bufferX))
        grid_y = int(height / (sizeY - bufferY))

        lastX = width % (sizeX - bufferX)
        lastY = height % (sizeY - bufferY)

        im_data = inDataset.ReadAsArray(0, 0, 20, 20)
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        split_jpg_count = (grid_x + 1) * (grid_y + 1)
        process_queue = multiprocessing.Queue()

        th_lst = self.start_mul_process(
            jpg_count=split_jpg_count,
            process_queue=process_queue
        )

        pict_cut_info = {}  # record the info of picture after cutting
        for j in range(grid_y + 1):
            if j == grid_y and lastY == 0:
                break
            startY = (sizeY - bufferY) * j
            lonY = startY * geo_info[5] + geo_info[3]

            for i in range(grid_x + 1):
                if i == grid_x and lastX == 0:
                    break
                startX = (sizeX - bufferX) * i
                lonX = startX * geo_info[1] + geo_info[0]

                # skip border
                if startY + sizeY > height or startX + sizeX > width:
                    continue

                outputFile = outputPath + '\\' + name + str(count) + '_' + str(startY) + '_' + str(startX) + '.tif'
                envolope = [lonX, lonY, geo_info[1], geo_info[5]]
                simpleName = outputFile.split('\\')[-1]
                pict_cut_info[simpleName] = envolope

                if startY + sizeY > height and startX + sizeX > width:
                    datas = inDataset.ReadAsArray(startX, startY, width - startX, height - startY, band_list=band_list).tobytes()
                elif startY + sizeY > height:
                    datas = inDataset.ReadAsArray(startX, startY, sizeX, height - startY, band_list=band_list).tobytes()
                elif startX + sizeX > width:
                    datas = inDataset.ReadAsArray(startX, startY, width - startX, sizeY, band_list=band_list).tobytes()
                else:
                    datas = inDataset.ReadAsArray(startX, startY, sizeX, sizeY, band_list=band_list).tobytes()

                obj = CSaveSplitJpg(
                    sizeX=sizeX,
                    sizeY=sizeY,
                    rasterCount=len(band_list),
                    datatype=datatype,
                    height=height,
                    width=width,
                    startX=startX,
                    startY=startY,
                    datas=datas,
                    outputFile=outputFile
                )

                process_queue.put(obj)

        for _ in th_lst:
            process_queue.put("end")

        for th in th_lst:
            th.join()
        count += 1
        return pict_cut_info, count

    def run_process_thread(self, process_queue):
        p = multiprocessing.Process(target=run_save_split_jpg,
                                    args=(process_queue,))
        p.daemon = True
        p.start()
        p.join()

    def start_mul_process(self, jpg_count, process_queue):
        process_count = 10
        if process_count > jpg_count:
            process_count = jpg_count

        th_lst = []
        for _ in range(process_count):
            th = threading.Thread(target=self.run_process_thread,
                                  args=(process_queue,))
            th.start()

            th_lst.append(th)

        return th_lst

    def process(self, inputfile, outputpath, sizeX=1200, sizeY=1200, bufferX=200, bufferY=200, count=1, sheets=True):
        start_time = time()
        (pic_cut_info, count) = self.imageCutByRasterIO(inputFile=inputfile,
                                                        outputPath=outputpath,
                                                        bufferX=bufferX,
                                                        bufferY=bufferY,
                                                        sizeX=sizeX,
                                                        sizeY=sizeY,
                                                        count=count,
                                                        sheets = sheets)

        #        g_log.info("PictCut process COST {} ms".format(int((time() - start_time) * 1000)))
        print("PictCut process COST {} ms".format(int((time() - start_time) * 1000)))
        print(pic_cut_info)
        return pic_cut_info, count


if __name__ == '__main__':

    train = ["6751","6885","6902","6905","6906","6910","6911","6915","6917"]
    test = ["6909"]
    #
    for id in train:
        PictCut().process(inputfile="D:\\SanbornMap\\modified_2\\"+ id+ ".tif",
                          outputpath=r"D:\SanbornMap\UNet_brick_frame\data_modified_0604\train\labels",
                          sizeX=256, sizeY=256, bufferX=128, bufferY=128, count=1, sheets=False)
        PictCut().process(inputfile="D:\\SanbornMap\\data\\train\\sheets\\" + id +".tif",
                          outputpath=r"D:\SanbornMap\UNet_brick_frame\data_modified_0604\train\sheets",
                          sizeX=256, sizeY=256, bufferX=128, bufferY=128, count=1)

        # PictCut().process(inputfile="D:\\SanbornMap\\modified_frame\\" + id + ".tif",
        #                   outputpath=r"D:\SanbornMap\UNet_family\data_frame\train\label",
        #                   sizeX=256, sizeY=256, bufferX=128, bufferY=128, count=1, sheets=False)
        # PictCut().process(inputfile="D:\\SanbornMap\\data\\train\\sheets\\" + id + ".tif",
        #                   outputpath=r"D:\SanbornMap\UNet_family\data_frame\train\sheet",
        #                   sizeX=256, sizeY=256, bufferX=128, bufferY=128, count=1)

    for id in test:
        PictCut().process(inputfile="D:\\SanbornMap\\modified_2\\"+ id+ ".tif",
                          outputpath=r"D:\SanbornMap\UNet_brick_frame\data_modified_0604\test\labels",
                          sizeX=256, sizeY=256, bufferX=0, bufferY=0, count=1, sheets=False)
        PictCut().process(inputfile="D:\\SanbornMap\\data\\test\\sheets\\" + id +".tif",
                          outputpath=r"D:\SanbornMap\UNet_brick_frame\data_modified_0604\test\sheets",
                          sizeX=256, sizeY=256, bufferX=0, bufferY=0, count=1)

