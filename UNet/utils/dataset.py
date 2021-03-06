from typing import Any
import os

import torch.utils.data
from osgeo import gdal
import numpy as np

from torchvision.datasets.vision import VisionDataset


class SanbornDataset(VisionDataset):

    def __init__(self, map_path, label_path):
        self.map_path = map_path
        self.label_path = label_path
        self.num_pairs = len(os.listdir(self.label_path))

    def __getitem__(self, index: int) -> Any:
        label_files = os.listdir(self.label_path)
        label_path_name = label_files[index]

        sheet_path_name = os.path.join(self.map_path, label_path_name)
        sheet = gdal.Open(sheet_path_name)

        band1 = sheet.GetRasterBand(1)  # Red channel
        band2 = sheet.GetRasterBand(2)  # Green channel
        band3 = sheet.GetRasterBand(3)  # Blue channel

        b1 = band1.ReadAsArray() / 255
        b2 = band2.ReadAsArray() / 255
        b3 = band3.ReadAsArray() / 255

        image = np.stack((b1, b2, b3), axis=2)
        image = np.transpose(image, (2, 0, 1))   # input is Channel, Height, Width

        label = gdal.Open(os.path.join(self.label_path, label_path_name))
        gt_label = label.GetRasterBand(1).ReadAsArray()
        # gt_label = np.where(gt_label==255,1,0)

        return image, gt_label

    def __len__(self) -> int:
        return self.num_pairs


if __name__ == "__main__":
    # test = SanbornDataset(map_path=r"..\\data\\tif",
    #                       label_path=r"..\\data\\label")
    # data_loader = torch.utils.data.DataLoader(test, batch_size=1)
    # for idx, (map, label) in enumerate(data_loader):
    #
    #     print(label)

    # label_dir = r"D:\SanbornMap\UNet_brick_frame\data\train\polygons"
    # i=0
    # #
    # # label = gdal.Open(img_tif)
    # # non_zero = np.count_nonzero(label)
    # for file in os.listdir(label_dir):
    #     file_path = os.path.join(label_dir, file)
    #     label = gdal.Open(file_path)
    #
    #     gt_label = label.GetRasterBand(1).ReadAsArray()
    #     label = None
    #     if(np.count_nonzero(gt_label)==0):
    #         i += 1
    #         # file_path.close()
    #         os.remove(file_path)
    #
    # print("removed " + str(i))

    gt_img_path = r"D:\SanbornMap\UNet_brick_frame\data\train\polygons\69061_3200_3712.tif"
    label = gdal.Open(gt_img_path)
    gt_label = label.GetRasterBand(1).ReadAsArray()
    print(gt_label)
