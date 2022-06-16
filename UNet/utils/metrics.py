import numpy as np
import torch


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT):
    corr = np.sum(SR == GT)
    size = SR.shape[0] * SR.shape[1]
    acc = float(corr) / float(size)

    return acc


def get_recall(SR, GT):
    # Sensitivity == Recall

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1) & (GT == 1))
    FN = ((SR == 0) & (GT == 1))

    SE = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)

    return SE


def get_precision(SR, GT):

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1) & (GT == 1))
    FP = ((SR == 1) & (GT == 0))

    PC = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT):
    # Sensitivity == Recall
    SE = get_recall(SR, GT)
    PC = get_precision(SR, GT)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT):
    # JS : Jaccard similarity

    Inter = np.sum((SR + GT) == 2)
    Union = np.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient

    Inter = np.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC