import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import h5py
import scipy.io as scio
from glob import glob
import SimpleITK as sitk
import random
import cv2

def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return  img, mask

class Prostate(Dataset):
    '''
    Six prostate dataset (BIDMC, HK, I2CVB, ISBI, ISBI_1.5, UCL)
    '''
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'BIDMC':3, 'HK':3, 'I2CVB':3, 'ISBI':3, 'ISBI_1.5':3, 'UCL':3}
        assert site in list(channels.keys())
        self.split = split
        
        base_path = base_path if base_path is not None else '/research/d6/gds/mrjiang/Dataset/Prostate/preprocessed'
        
        with open(os.path.join(base_path,site+f'_{split}.txt'),'r') as f:
            f_names = [line.rstrip() for line in f.readlines()]
        images, labels = [], []
        for f_name in f_names:
            a = np.load(os.path.join(base_path,site,f_name))
            images.append(np.load(os.path.join(base_path,site,f_name)))
            labels.append(np.load(os.path.join(base_path,site,f_name.replace('.npy','_segmentation.npy'))))
        images = np.concatenate(images,axis=0)
        labels = np.concatenate(labels,axis=0)
        
        labels = np.array(labels).astype(int)
        
        
        self.images = images
        self.labels = labels

        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        '''
        if self.split == 'train':
            R1 = RandomRotate90()
            image, label = R1(image, label)
            R2 = RandomFlip()
            image, label = R2(image, label)
        '''
        image = np.transpose(image,(2, 0, 1))
        image = torch.Tensor(image)
        label = torch.Tensor(label)[None,:]

        return image, label


if __name__=='__main__':
    pass


