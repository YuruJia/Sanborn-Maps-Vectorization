import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F
import torch
from torchvision import transforms


def plot_img_and_mask(img, mask, gt, save_path):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 2:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Ground Truth')
        ax[1].imshow(gt)
        ax[2].set_title(f'Output Mask')
        ax[2].imshow(mask[1, :, :])
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path)
    plt.show()



def plot_loss_peak(img, step):
    img = img.cpu().detach().numpy()

    for i in range(img.shape[0]):
        img_i = img[i]
        img_i = np.transpose(img_i, (1, 2, 0))

        plt.imshow(img_i)
        plt.show()
        save_path = 'loss-peak-images/'+str(step)+'_'+str(i)+".png"
        plt.savefig(save_path)
