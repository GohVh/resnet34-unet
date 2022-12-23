import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import pandas as pd

def create_df(path):
        name = []
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                name.append(filename.split('.')[0])        
        return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

def plot(history, graphType, isTest=False):
    if not isTest:
        plt.plot(history[f'train_{graphType}'], label='train', marker= '*')
        plt.plot(history[f'val_{graphType}'], label='val', marker = 'o')
    else:
        plt.plot(history[f'test_{graphType}'], label='test', marker= '*')
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

def load_train_config(path):
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

def save_checkpoint(state, current_checkpoint_path, is_best=False, best_model_path=None):
    torch.save(state, current_checkpoint_path)
    if is_best:
        assert best_model_path!=None, 'best_model_path should not be None.'
        shutil.copyfile(current_checkpoint_path, best_model_path)

def load_checkpoint(best_model_path, current_checkpoint_path, model, optimizer, scheduler, best_checkpoint=False):
    
    train_loss_key = 'train_loss'
    val_loss_key = 'val_loss'
    path = current_checkpoint_path

    if best_checkpoint:
        path = best_model_path
        
    model, optimizer, scheduler, epoch, train_loss, val_loss = load_model(path, model, optimizer, scheduler, train_loss_key, val_loss_key)    
    print(f'optimizer = {optimizer}, start epoch = {epoch}, train loss = {train_loss}, val loss = {val_loss}')
    return model, optimizer, scheduler, val_loss

def load_model(model_path, model, optimizer, scheduler, train_loss_key, val_loss_key):
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint[train_loss_key]
    val_loss = checkpoint[val_loss_key]
    
    return model, optimizer, scheduler, epoch, train_loss, val_loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
