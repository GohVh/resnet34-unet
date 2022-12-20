from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision

from torch.autograd import Variable
import albumentations as A
import os

from torchsummary import summary

class DroneDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, datasetType, mean=None, std=None, transform=False, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.datasetType = datasetType
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def _transfrom(self):
        if self.datasetType=='TRAIN':
            t = A.Compose(
                [A.Resize(768, 1024,
                interpolation=cv2.INTER_NEAREST), 
                A.HorizontalFlip(), 
                A.VerticalFlip(),
                A.GridDistortion(p=0.2),
                A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                A.GaussNoise()])
        
        elif self.datasetType=='VAL':
            t = A.Compose(
                [A.Resize(768, 1024,
                interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(),
                A.GridDistortion(p=0.2)])
        
        elif self.datasetType=='TEST':
            t = A.Resize(768, 1024, interpolation=cv2.INTER_NEAREST)
        
        return t
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            aug = self._transfrom()(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        else:
            img = Image.fromarray(img)
        
        if self.datasetType=='TRAIN' or self.datasetType=='VAL':
            img = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])(img)

        mask = torch.from_numpy(mask).long()
        
        if self.patches:
            img, mask = self.tiles(img, mask)
            
        return img, mask
    
    def tiles(self, img, mask):

        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768) 
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768) 
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        
        return img_patches, mask_patches
