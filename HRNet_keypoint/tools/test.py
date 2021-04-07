# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils import data as data_utils
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.TestDataset import TestDataset

from sklearn.model_selection import train_test_split

import time
import pickle
import dataset
import models
import numpy as np
import pandas as pd
import pdb
import random
from pathlib import Path
from core.inference import get_final_preds

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    
    parser.add_argument('--output_path',
                        help="output path",
                        required=True,
                        default=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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




def main():
    args = parse_args()
    update_config(cfg, args)


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    
    data_path = './data'
    batch_size = cfg.TEST.BATCH_SIZE_PER_GPU
    output_path = cfg.output_path
    
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(output_path+'/model_best.pth'))
    
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    imgs = os.listdir(f'{data_path}/images/test_imgs/')
    
    test_dataset = TestDataset(cfg=cfg, root=data_path, image_set = imgs,
                         is_train = False, phase='test',
                         transform= transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                                 ]))
    
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    # evaluate on validation set
    
    y_pred = np.array([])

    #y_box = np.array([])
    filenames = []
    with torch.no_grad():
        
        
        for i, (input,  meta) in enumerate(test_loader):
            # compute output
            input = input.to(device)
            input = input.float()
    
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

           #num_images = input.size(0)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
    
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), c, s)
    
    
            preds = preds.reshape(preds.shape[0],-1)
    
            if len(y_pred) == 0:
                y_pred = preds

            else :
                y_pred = np.r_[y_pred, preds]

            filenames += meta['filename']

    df_sub = pd.read_csv(f'{data_path}/sample_submission.csv')
    df = pd.DataFrame(columns=df_sub.columns)
    df['image'] = filenames
    df.iloc[:, 1:] = y_pred
    df.head()

    df.to_csv(f'{output_path}/result.csv', index=False)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
