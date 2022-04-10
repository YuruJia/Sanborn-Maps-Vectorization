#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from crop_tif import PictCut
from osgeo import gdal
from osgeo import ogr
import shutil
import cv2
import numpy as np

class_name = ['water', 'tree', 'building', 'water', 'road']


class imgProcess(object):

    def getRect(self, point_ul, point_lr):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(point_ul[0], point_ul[1])
        ring.AddPoint(point_lr[0], point_ul[1])
        ring.AddPoint(point_lr[0], point_lr[1])
        ring.AddPoint(point_ul[0], point_lr[1])
        ring.CloseRings()
        rect = ogr.Geometry(ogr.wkbPolygon)
        rect.AddGeometry(ring)
        return rect

    def getGeometries(self, layer, point_ul, point_lr, gsd_x, gsd_y):
        layer.SetSpatialFilterRect(point_ul[0], point_ul[1], point_lr[0], point_lr[1])
        oFeature = layer.GetNextFeature()
        oDefn = layer.GetLayerDefn()
        iFieldCount = oDefn.GetFieldCount()
        geometry_list = []

        while oFeature is not None:
            """
            value = None
            
            for iField in range(iFieldCount):
                oFieldDefn = oDefn.GetFieldDefn(iField)
                if not (oFieldDefn.GetNameRef().lower() in ['class_id', 'id', 'feat_type',
                                                            'faet_type']):  # modified to read class id
                    continue
                value = oFeature.GetFieldAsString(iField)
                break
            if value != '8007':  # right value should be 1 - 5
                oFeature = layer.GetNextFeature()
                continue
            """
            # geometry_list.append([oFeature, class_name[int(value) - 1]])
            geometry_list.append([oFeature, "building"])
            oFeature = layer.GetNextFeature()

        rect = self.getRect(point_ul, point_lr)
        geometry_list_intersection = []

        for fea in geometry_list:
            polygon = fea[0].GetGeometryRef()
            polygon_union = polygon
            if polygon.GetGeometryCount() > 1:
                for ring in fea[0].GetGeometryRef():
                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)
                    polygon_union = polygon_union.Union(poly)

            geometry_intersection = rect.Intersection(polygon_union)
            geometry_list_intersection.append([geometry_intersection, fea[1]])

        # for geometry in geometry_list:
        #     geometry_intersection = rect.Intersection(geometry[0].GetGeometryRef())
        #     geometry_list_intersection.append([geometry_intersection, geometry[1]])
        origin_x = point_ul[0]
        origin_y = point_ul[1]
        geometries = []
        for geometry in geometry_list_intersection:
            if geometry[0] is None:
                continue
            strval = geometry[0].ExportToWkt().split('((')[1]
            pts_str = strval[0:len(strval) - 2].split(',')
            pts = []
            for i in range(len(pts_str)):
                pt_str = pts_str[i].split(' ')
                if pt_str[0].startswith('('):
                    pt_str[0] = pt_str[0][1:]
                if pt_str[1].endswith(')'):
                    pt_str[1] = pt_str[1][:-1]
                # print('--------', pt_str, origin_x, origin_y)
                pt = [(float(pt_str[0]) - origin_x) / gsd_x, (float(pt_str[1]) - origin_y) / gsd_y]
                pts.append(pt)
            pts.append(geometry[1])
            geometries.append(pts)
        return geometries

    def writeJson(self, geometries, img, output_json_path, jpg_file_source):
        pict_info = dict()
        pict_info['image'] = img
        pict_info['source_name'] = jpg_file_source
        objects = []
        for geometry in geometries:
            polygon_info = dict()
            polygon_info['polygon'] = geometry[:-1]

            # if geometry[-1] == '' or geometry[-1] is None:
            #     geometry[-1] = 'light_pole'
            # import re
            # pattern = re.compile(r'ligh.*')
            # if re.search(pattern, geometry[-1]):
            #     geometry[-1] = 'light_pole'
            polygon_info['label'] = geometry[-1]
            objects.append(polygon_info)
            pict_info['objects'] = objects

        json_file = img.split('.')[0] + '.json'
        json_path = os.path.join(output_json_path, json_file)
        f = open(json_path, 'w')
        json.dump(pict_info, f)

    def draw_polygon(self, ref_img_dir, output_dir, filename, geometries, fill=False):
        ref_jpg_file = os.path.join(ref_img_dir, filename)
        if fill:
            img = cv2.imread(ref_jpg_file, -1)
            frame = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
            img = None
        else:
            frame = cv2.imread(ref_jpg_file, -1)
        # print(geometries)
        for i in range(len(geometries)):
            node_polygon = []
            for j in range(len(geometries[i]) - 1):
                pt = (geometries[i][j][0], geometries[i][j][1])
                node_polygon.append(pt)
            Pts = np.array(node_polygon, np.int32)
            # print('------', node_polygon)
            # print('-----------', Pts)
            if fill:
                cv2.fillPoly(frame, [Pts], 255)
            else:
                cv2.polylines(frame, [Pts], True, (255, 255, 0), 5)
        cv2.imwrite(os.path.join(output_dir, filename), frame)

    def file_check(self, rootdir):
        if os.path.exists(rootdir):
            files = os.listdir(rootdir)
            for file in files:
                file_path = os.path.join(rootdir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path, True)
        else:
            os.makedirs(rootdir)

    def process(self, input_tif_dir, input_shp_dir, output_dir, sizeX, sizeY, bufferX, bufferY):

        gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'YES')
        gdal.SetConfigOption('SHAPE_ENCODING', '')

        ogr.RegisterAll()

        output_tif_dir = os.path.join(output_dir, "tif")
        self.file_check(output_tif_dir)

        output_json_dir = os.path.join(output_dir, "json")
        self.file_check(output_json_dir)

        output_label_dir = os.path.join(output_dir, "label")
        self.file_check(output_label_dir)

        output_check_dir = os.path.join(output_dir, "check")
        self.file_check(output_check_dir)
        count = 1

        driver = ogr.GetDriverByName('ESRI Shapefile')
        for shp_file in os.listdir(input_shp_dir):
            shp_path = os.path.join(input_shp_dir, shp_file)

            if not shp_file.endswith('.shp') and not shp_file.endswith('.dbf') and not shp_file.endswith('.shx'):
                os.remove(shp_path)
                continue
            if not shp_file.endswith('.shp'):
                continue
            shp_path = os.path.join(input_shp_dir, shp_file)
            ds = driver.Open(shp_path)
            if ds is None:
                print(ds, "can't be openned!")
            layer = ds.GetLayer()
            fea = layer.GetNextFeature()
            if fea is None:
                ds.Destroy()
                os.remove(shp_path)
                os.remove(shp_path.replace('.shp', '.shx'))
                os.remove(shp_path.replace('.shp', '.dbf'))
                continue

            findImg = False
            for ext in [".tif", ".img", ".jpg"]:
                tif_file = shp_file.replace('.shp', ext)
                input_tif_path = os.path.join(input_tif_dir, tif_file)
                if os.path.isfile(input_tif_path):
                    findImg = True
                    break
            if not findImg:
                print("%s can't find!" % input_tif_path)
                continue

            print(shp_file + '\t' + tif_file)
            pic_cutter = PictCut()
            (pic_cut_info, count) = pic_cutter.process(input_tif_path, output_tif_dir, sizeX, sizeY, bufferX,
                                                       bufferY, count)

            for key, value in pic_cut_info.items():
                if value[3] > 0.0:
                    value[3] = -value[3]
                    point_ul = [float(value[0]), -float(value[1])]
                    point_lr = [float(value[0]) + float(value[2]) * sizeX, -float(value[1]) + float(value[3]) * sizeY]
                else:
                    point_ul = [float(value[0]), float(value[1])]
                    point_lr = [float(value[0]) + float(value[2]) * sizeX, float(value[1]) + float(value[3]) * sizeY]
                geometries = self.getGeometries(layer, point_ul, point_lr, float(value[2]), float(value[3]))
                if len(geometries) > 0:
                    print('Write to json :', key)
                    self.draw_polygon(output_tif_dir, output_label_dir, key, geometries, True)
                    self.draw_polygon(output_tif_dir, output_check_dir, key, geometries)
                    self.writeJson(geometries, key, output_json_dir, tif_file)
            print('================================================')
        print('###############    All Done!     ################')


if __name__ == '__main__':


    imgProcess().process(input_tif_dir=r"D:\SanbornMap\data\sheets",
                         input_shp_dir=r"D:\SanbornMap\data\buildings",
                         output_dir=r"..\data",
                         sizeX=512, sizeY=512, bufferX=256, bufferY=256)

