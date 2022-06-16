import os
from osgeo import gdal
import numpy as np

def del_black_label(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        label = gdal.Open(file_path)

        gt_label = label.GetRasterBand(1).ReadAsArray()
        label = None
        if np.count_nonzero(gt_label)==0:
            os.remove(file_path)


if __name__ == "__main__":
    path1 = r"D:\SanbornMap\UNet_brick_frame\data_modified_0604\train\labels"
    del_black_label(path1)