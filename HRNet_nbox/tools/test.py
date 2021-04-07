from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import _init_paths
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch.utils import data as data_utils

from config import cfg
from config import update_config

from utils.utils import create_logger

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.dataset import TestDataset
import models
import pickle

import numpy as np
import pandas as pd
import pdb
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    
    parser.add_argument('--output_path',
                        help="output path",
                        default=str,
                        nargs=argparse.REMAINDER)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args



class Split(torch.nn.Module):
    def __init__(self, module, parts: int, dim=1):
        super().__init__()
        self.parts
        self.dim = dim
        self.module = module

    def forward(self, inputs):
        output = self.module(inputs)
        chunk_size = output.shape[self.dim] // self.parts
        return torch.split(output, chunk_size, dim=self.dim)




class Mynet(nn.Module):
    def __init__(self, cfg):
        super(Mynet, self).__init__()
        
        num_classes = cfg['MODEL']['NUM_JOINTS']
        extra = cfg['MODEL']['EXTRA']
        BN_MOMENTUM = 0.1

        
        self.final_conv = nn.Conv2d(
            in_channels = num_classes,
            out_channels = num_classes,
            kernel_size = 2,
            stride=2,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )
        
        if cfg['MODEL']['IMAGE_SIZE'][1] == 384 :
            if cfg['MODEL']['IMAGE_SIZE'][1] == cfg['MODEL']['IMAGE_SIZE'][0] :
                self.bbox_layer = nn.Linear(num_classes*72*72, 4)
                self.keypoints_layer = nn.Linear(num_classes*72*72, 48)
            else :
                self.bbox_layer = nn.Linear(num_classes*72*96, 4)
                self.keypoints_layer = nn.Linear(num_classes*72*96, 48)
            
        elif cfg['MODEL']['IMAGE_SIZE'][1] == 256 :
            self.bbox_layer = nn.Linear(num_classes*64*64, 4)
            self.keypoints_layer = nn.Linear(num_classes*64*64, 48)
        else :
            self.bbox_layer = nn.Linear(num_classes*48*48, 4)
            self.keypoints_layer = nn.Linear(num_classes*48*48, 48)
        
        self.bn1 = nn.BatchNorm2d(num_classes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        """
        x = self.final_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        """
        x = x.view(x.shape[0],-1)
        
        b_y = self.bbox_layer(x)
        """
        k_y = self.keypoints_layer(x)
        """
        return b_y
        


# https://github.com/amdegroot/pytorch-containers/blob/master/README.md
def initialize_model(model_ft, cfg):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.

    mynet = Mynet(cfg)
    
    torch.nn.init.xavier_uniform_(mynet.final_conv.weight)
    torch.nn.init.xavier_uniform_(mynet.keypoints_layer.weight)
    torch.nn.init.xavier_uniform_(mynet.bbox_layer.weight)
    
    f_model = nn.Sequential(
            model_ft,
            mynet
            )
    
    
    return f_model



def main():
    
    data_path = './data'
    args = parse_args()
    update_config(cfg, args)
    
    #data_path = './data'
    batch_size = cfg.TEST.BATCH_SIZE_PER_GPU
    final_output_dir = cfg.output_path

    input_w = cfg.MODEL.IMAGE_SIZE[1]
    input_h = cfg.MODEL.IMAGE_SIZE[0]
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    RANDOM_SEED = int(cfg.RANDOMSEED)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    # model 수정
    
    model = initialize_model(model, cfg)
    
    # copy model file
  
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.load_state_dict(torch.load(final_output_dir+'/model_best.pth'))
    #logger.info(get_model_summary(model, dump_input.cuda()))
    
    # define loss function (criterion) and optimizer
    
    # Data Augumentation
    random.seed(RANDOM_SEED)
    A_transforms = {
    'train':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.OneOf([A.HorizontalFlip(p=1),
                     A.VerticalFlip(p=1),
            ], p=0.5),
            A.OneOf([A.MotionBlur(p=1),
                     A.GaussNoise(p=1)                 
            ], p=0.5),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True, angle_in_degrees=True)),
    
    'val':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True, angle_in_degrees=True)),
    
    'test':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    }
    
    
    # Data loading code
        # parameter setting
        
    batch_size = 32
    test_dir = f'{data_path}/images/test_imgs'
    test_imgs = os.listdir(test_dir)
    
    
    test_data = TestDataset(test_dir, test_imgs, data_transforms=A_transforms, phase='test')
    test_loader = data_utils.DataLoader(test_data, batch_size=batch_size * 4, shuffle=False)
    
    
    all_predictions = []
    files = []
    
    width = 1920
    height = 1080

    with torch.no_grad():
        for filenames, inputs in test_loader:
            
            result = model(inputs.to(device))
            #_, result = model_ft(inputs.to(device))
            predictions = list(result.cpu().numpy())
            files.extend(filenames)

            for prediction in predictions:
                all_predictions.append(prediction)
                
    all_predictions = np.array(all_predictions)
    all_predictions[:, 0] /= input_w / 1920
    all_predictions[:, 1] /= input_h / 1080
    all_predictions[:, 2] /= input_w / 1920
    all_predictions[:, 3] /= input_h / 1080
    
    results = []
    for i in range(len(files)) :
        min_x, min_y, max_x, max_y = all_predictions[i,:]
        filename = files[i]
        w = abs(max_x - min_x)
        h = abs(max_y - min_y)
        area = w * h
        temp_dict = {'width':width, 'height':height, 'area':area, 
                        'bbox':[min_x, min_y, w, h], 'filename':filename}

        results.append(temp_dict)
        
    with open(final_output_dir+'/test_annotation.pkl', 'wb') as f:
        pickle.dump(results, f)
    
     

if __name__ == '__main__':
    main()