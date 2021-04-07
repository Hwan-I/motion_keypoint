#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 02:51:59 2021

@author: lch
"""
import cv2
from torch.utils import data as data_utils
import numpy as np
import os

import pdb
    


class Dataset(data_utils.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, data_dir, imgs, bbox, phase, class_labels=None, data_transforms=None):
        self.data_dir = data_dir
        self.imgs = imgs
        self.bbox = bbox
        self.phase = phase
        self.class_labels = class_labels
        self.data_transforms = data_transforms


    def __getitem__(self, idx):
        # Read an image with OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.imgs[idx]))
        bbox = self.bbox[idx]
        
        # transform이 있을 경우 이를 적용합니다.
        if self.data_transforms:
            augmented = self.data_transforms[self.phase](image=img, bboxes=bbox, class_labels=self.class_labels)
            img = augmented['image']
            bbox = augmented['bboxes']
        
        # bbox 객체를 정리합니다.
            # 형태 : [min_x, min_y, max_x, max_y]
        bbox = list(bbox[0])

        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox = np.array(bbox)
        
        if type(img) != np.ndarray :
            img = img.numpy()
        
        return img, bbox
    
    def __len__(self):
        return len(self.imgs)



class TestDataset(data_utils.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, data_dir, imgs, phase, data_transforms=None):
        self.data_dir = data_dir
        self.imgs = imgs
        self.phase = phase
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        # Read an image with OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.imgs[idx]))

        if self.data_transforms:
            augmented = self.data_transforms[self.phase](image=img)
            img = augmented['image']
        return filename, img
    
    def __len__(self):
        return len(self.imgs)