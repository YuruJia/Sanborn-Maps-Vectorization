import argparse
import logging
import os

import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.plot_img_and_mask import plot_img_and_mask
import pandas as pd

from osgeo import gdal

from model.networks import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from config import config
from utils.metrics import *



model_type = config["test-model-type"]
in_chn = config["in_chn"]
out_nclass = config["out_nclass"]
t = config["recurrent_t"]
test_ckp_path = config["test-checkpoint-path"]
test_img_dir = config["test-img-dir"]
test_gt_dir = config["test-gt-dir"]

batch_size = config["batch_size"]



multiclass = True if out_nclass > 1 else False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def read_test_image(img_path):
    img = gdal.Open(img_path)

    band1 = img.GetRasterBand(1)  # Red channel
    band2 = img.GetRasterBand(2)  # Green channel
    band3 = img.GetRasterBand(3)  # Blue channel

    b1 = band1.ReadAsArray() / 255
    b2 = band2.ReadAsArray() / 255
    b3 = band3.ReadAsArray() / 255

    image = np.stack((b1, b2, b3), axis=2)
    image = np.transpose(image, (2, 0, 1))

    return image


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(full_img).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.output_chn > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.shape[1], full_img.shape[2])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.output_chn == 1:
        return (full_mask > out_threshold).numpy()
    else:
        #return F.one_hot(full_mask.argmax(dim=0), net.output_chn).permute(2, 0, 1).numpy()
        return full_mask.argmax(dim=0).numpy()



def mask_to_image(mask: np.ndarray):
    return Image.fromarray(mask.astype(np.uint8))
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
        #return Image.fromarray(mask.astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def predict():
    if model_type == 'U_Net':
        unet = U_Net(img_ch=in_chn, output_ch=out_nclass)
    elif model_type == 'R2U_Net':
        unet = R2U_Net(img_ch=in_chn, output_ch=out_nclass, t=t)
    elif model_type == 'AttU_Net':
        unet = AttU_Net(img_ch=in_chn, output_ch=out_nclass)
    elif model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=in_chn, output_ch=out_nclass, t=t)

    logging.info(f'Loading model {test_ckp_path}')
    logging.info(f'Using device {device}')

    unet.to(device=device)
    unet.load_state_dict(torch.load(test_ckp_path, map_location=device))

    logging.info('Model loaded!')

    acc = np.array([])
    prec = np.array([])
    recall = np.array([])
    f1 = np.array([])
    iou = np.array([])

    save_img = True
    vis = False
    confusion_matirx = torch.zeros(out_nclass, out_nclass,dtype=torch.int64)
    for i, filename in enumerate(os.listdir(test_img_dir)):
        test_img_path = os.path.join(test_img_dir, filename)
        img = read_test_image(test_img_path)

        mask = predict_img(net=unet,
                           full_img=img,
                           device=device)

        gt_file = os.path.join(test_gt_dir, filename)
        label = gdal.Open(gt_file)
        gt_label = label.GetRasterBand(1).ReadAsArray()

        stacked = np.stack((gt_label.reshape(-1,1),mask.reshape(-1,1)),axis=1).squeeze()
        for p in stacked:
            ll,ml = p.tolist()
            confusion_matirx[ll,ml] = confusion_matirx[ll,ml] + 1



        # if(os.path.exists(gt_file)):
        #     label = gdal.Open(gt_file)
        #     gt_label = label.GetRasterBand(1).ReadAsArray()
        #     #gt = np.where(gt_label == 255, 1, 0)
        # else: # the file doesn't exist, meaning that no buildings on the image
        #     gt = np.zeros(shape=(img.shape[1], img.shape[2]))

        """
        acc = np.append(acc, get_accuracy(mask[1], gt))

        if 1 in gt:
            prec = np.append(prec, get_precision(mask[1], gt))
            recall = np.append(recall, get_recall(mask[1], gt))
            f1 = np.append(f1, get_F1(mask[1], gt))
            iou = np.append(iou, get_JS(mask[1], gt))
        """
        if save_img:
            out_filename = "pred_0605\\" + filename
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if vis:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            img = np.transpose(img,(1, 2, 0))
            save_path = f'.\\test_output\\{model_type}_{filename}.png'
            plot_img_and_mask(img, mask,save_path)


    # with open("test_metrics.txt", 'a') as f:
    #     f.write(model_type + "\n")
    #     f.write("Accuracy: " + str(acc.mean()) + "\n")
    #     f.write("Precision: " + str(prec.mean()) + "\n")
    #     f.write("Recall: "  + str(recall.mean()) + "\n")
    #     f.write("F1 score: "  + str(f1.mean()) + "\n")
    #     f.write("IoU score: " + str(iou.mean()) + "\n")
    #
    label_names = ["non building", "brick", "frame"]
    confmat_df = pd.DataFrame(data=confusion_matirx, columns=label_names, index=label_names)
    confmat_df.to_csv("confusion_matrix.csv")


if __name__ == "__main__":
    predict()