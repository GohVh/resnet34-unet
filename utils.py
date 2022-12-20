import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test import *

def dummyfucntion():
    return True

def plot(history, graphType):
        plt.plot(history[f'train_{graphType}'], label='train', marker= '*')
        plt.plot(history[f'val_{graphType}'], label='val', marker = 'o')
        plt.title(f'{graphType} per epoch')
        plt.ylabel(f'{graphType}')
        plt.xlabel('epoch')
        plt.legend(), plt.grid()
        plt.show()


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, n_classes=24, smooth=1e-10):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def load_train_config(path="config.yaml"):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data

def visualize(image, mask, pred_mask, score):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
    ax1.imshow(image)
    ax1.set_title('Picture')

    ax2.imshow(mask)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(pred_mask)
    ax3.set_title('UnetResnet34 | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()