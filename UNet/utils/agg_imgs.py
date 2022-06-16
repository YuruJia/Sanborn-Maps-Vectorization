import os

from PIL import Image
import numpy as np

def agg_imgs(img_folder,img_id, offsetX, offsetY):
    x = []
    y = []
    for img_file in os.listdir(img_folder):
        img_num = img_file.split(".")[0]
        x.append(int(img_num.split("_")[2]))
        y.append(int(img_num.split("_")[1]))
    x_max = max(x)
    y_max = max(y)
    last_img = np.asarray(Image.open(os.path.join(img_folder,f"{img_id}_{y_max}_{x_max}.tif")))
    img_height = last_img.shape[0]+y_max
    img_width = last_img.shape[1]+x_max

    v_img = np.empty((0,img_width))
    for i in np.arange(0,y_max+1,offsetY):
        h_img = np.empty((offsetY,0))
        for j in np.arange(0,x_max+1,offsetX):
            cur_img = np.asarray(Image.open(os.path.join(img_folder,f"{img_id}_{i}_{j}.tif")))
            h_img = np.hstack((h_img,cur_img))
        print(h_img.shape)
        print(v_img.shape)
        v_img = np.vstack((v_img,h_img))

    res = Image.fromarray(v_img)
    save_path = "D:\\SanbornMap\\UNet_brick_frame\\" + "agg0605.tif"
    res.save(save_path)




if __name__ == "__main__":
    agg_imgs("D:\\SanbornMap\\UNet_brick_frame\\pred_0605\\", img_id=69091, offsetX=256, offsetY=256)

