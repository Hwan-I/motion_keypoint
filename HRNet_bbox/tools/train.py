# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

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

from dataset.dataset import Dataset, TestDataset
from dataset.copy_paste import CopyPaste
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    
    parser.add_argument('--test_option',
                        help='whether test data or not',
                        required=True,
                        type=str)


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




def make_train_data(data_path):
    train_df = pd.read_csv(data_path+'/train_df.csv')
    train_df = train_df.reset_index(drop=True)
    
    
    with open(data_path+'/annotations/train_annotation.pkl', 'rb') as f:
        annotations = pickle.load(f)
    
    anno_df = pd.DataFrame()
    filenames = []
    min_xs, min_ys = [], []
    ws, hs = [], []
    
    for obj in annotations :
        filename = obj['filename']
        bbox = obj['bbox']
        min_x, min_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        
        filenames.append(filename)
        min_xs.append(min_x)
        min_ys.append(min_y)
        ws.append(w)
        hs.append(h)
    
    anno_df['image'] = filenames
    anno_df['min_x'] = min_xs
    anno_df['min_y'] = min_ys
    anno_df['width'] = ws
    anno_df['height'] = hs
    #anno_df['class_labels'] = 'person'
    
    train_df = train_df.iloc[:, 0]
    train_df = pd.merge(train_df, anno_df, on='image', how='left')
    
    
    delete_list = [317, 869, 873, 877, 911, 1559, 1560, 1562, 1566, 1575, 
                   1577, 1578, 1582, 1606, 1607, 1622, 1623, 1624, 1625, 
                   1629, 3968, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 
                   4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 
                   4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 
                   4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 
                   4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 
                   4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 
                   4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 
                   4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 
                   4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194,
                  1597, 3864, 1216]
    
    train_df = train_df.drop(delete_list, axis=0).reset_index(drop=True)
    
    imgs = train_df.iloc[:, 0].to_numpy()
    
    bbox = train_df.iloc[:, 1:].values.tolist()
    class_labels = ['person']
    for i, box in enumerate(bbox) :
        bbox[i] = [box]
 

    

    return imgs, bbox, class_labels


def make_annotations(data_path) :
    key_df = pd.read_csv(data_path+'/train_df.csv')
    width = 1920
    height = 1080
    result_list = []
    
    for i in range(len(key_df)) :
        keypoints = key_df.iloc[i, 1:].values.reshape(-1,2)
        filename = key_df.iloc[i,0]
        min_x, min_y = np.min(keypoints, 0)
        max_x, max_y = np.max(keypoints, 0)
        w, h = max_x - min_x, max_y - min_y
        area = w * h

        keypoints = keypoints.flatten().tolist()
        temp_dict = {'width':width, 'height':height, 'keypoints':keypoints, 'area':area, 
                    'bbox':[min_x, min_y, w, h], 'filename':filename}
        result_list.append(temp_dict)
    
    with open(data_path+'/annotations/train_annotation.pkl', 'wb') as f:
        pickle.dump(result_list, f)

    


def main():
    
    data_path = './data'
    train_dir = Path(data_path, 'images/train_imgs')
    args = parse_args()
    update_config(cfg, args)
    
    lr = cfg.TRAIN.LR
    lamb = cfg.LAMB
    test_option = eval(cfg.test_option)
    
    input_w = cfg.MODEL.IMAGE_SIZE[1]
    input_h = cfg.MODEL.IMAGE_SIZE[0]
    

    RANDOM_SEED = int(cfg.RANDOMSEED)
    
    # https://hoya012.github.io/blog/reproducible_pytorch/
    
    np.random.seed(RANDOM_SEED) # cpu vars
    torch.manual_seed(RANDOM_SEED) # cpu  vars
    random.seed(RANDOM_SEED) # Python
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED) # Python hash buildin
    
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    #torch.use_deterministic_algorithms = True
    #torch.set_deterministic(True)
    
    
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, f'lr_{str(lr)}', 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # make_annotations
    if os.path.isfile(data_path+'/annotations/train_annotation.pkl') == False :
        make_annotations(data_path)
    
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    
    # model 수정
    
    model = initialize_model(model, cfg)
    
    
    
    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    
    shutil.copy2(
        os.path.join(this_dir, '../tools', 'train.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    
    
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    #logger.info(get_model_summary(model, dump_input.cuda()))
    
    # define loss function (criterion) and optimizer
    #criterion = nn.SmoothL1Loss().cuda()
    criterion = nn.MSELoss().cuda()
    

    """
    # Data Augumentation
    A_transforms = {
        
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
        
    if input_h == input_w :
        
        A_transforms['train'] = A.Compose([
                A.Resize(input_h, input_w, always_apply=True),
                A.OneOf([A.HorizontalFlip(p=1),
                         A.VerticalFlip(p=1),
                         A.Rotate(p=1),
                         A.RandomRotate90(p=1)
                ]),
                A.OneOf([A.MotionBlur(p=1),
                         A.GaussNoise(p=1)                 
                ], p=0.5),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True, angle_in_degrees=True))
        
    else :
        A_transforms['train'] = A.Compose([
                A.Resize(input_h, input_w, always_apply=True),
                A.OneOf([A.HorizontalFlip(p=1),
                         A.VerticalFlip(p=1),
                ]),
                A.OneOf([A.MotionBlur(p=1),
                         A.GaussNoise(p=1)                 
                ], p=0.5),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True, angle_in_degrees=True))
    """
    
    
    # https://albumentations.ai/docs/examples/example_bboxes/
    # https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/
    # pip install -U git+https://github.com/albumentations-team/albumentations
    # Data Augumentation
    A_transforms = {
        
        'val':
            A.Compose([
                A.Resize(input_h, input_w, always_apply=True),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05, label_fields=['class_labels'])),
        
        'test':
            A.Compose([
                A.Resize(input_h, input_w, always_apply=True),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        }
        
    if input_h == input_w :
        
        A_transforms['train'] = A.Compose([
                A.Resize(input_h, input_w, always_apply=True),
                A.OneOf([A.HorizontalFlip(p=1),
                         A.VerticalFlip(p=1),
                         A.Rotate(p=1),
                         A.RandomRotate90(p=1)
                ], p=0.5),
                A.OneOf([A.MotionBlur(p=1),
                         A.GaussNoise(p=1),
                         A.ColorJitter(p=1)
                ], p=0.5),

                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05, label_fields=['class_labels']))
        
    else :
        A_transforms['train'] = A.Compose([
                A.Resize(input_h, input_w, always_apply=True),
                A.OneOf([A.HorizontalFlip(p=1),
                         A.VerticalFlip(p=1),
                         A.Rotate(p=1),
                ], p=0.5),
                A.OneOf([A.MotionBlur(p=1),
                         A.GaussNoise(p=1)
                         
                ], p=0.5),
                A.OneOf([A.CropAndPad(percent=0.1, p=1),
                         A.CropAndPad(percent=0.2, p=1),
                         A.CropAndPad(percent=0.3, p=1)
                ], p=0.5),

                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05, label_fields=['class_labels']))
    
    # Data loading code
        # parameter setting
    batch_size = int(cfg.TRAIN.BATCH_SIZE_PER_GPU)
    test_ratio = float(cfg.TEST_RATIO)
    
    imgs, bbox, class_labels = make_train_data(data_path)
    num_epochs = cfg.TRAIN.END_EPOCH
    num_earlystop = num_epochs
    
    
    since = time.time()
    if test_option == True :
        X_train, X_test, y_train, y_test = train_test_split(imgs, bbox, test_size=0.1, random_state=RANDOM_SEED)
        test_dataset = [X_test, y_test]
        with open(final_output_dir+'/test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_ratio, random_state=RANDOM_SEED)
        test_data = Dataset(train_dir, X_test, y_test, data_transforms=A_transforms, class_labels=class_labels, phase='val')
        test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    else :
        X_train, X_val, y_train, y_val = train_test_split(imgs, bbox, test_size=test_ratio, random_state=RANDOM_SEED)
        
    train_data = Dataset(train_dir, X_train, y_train, data_transforms=A_transforms, class_labels=class_labels, phase='train')
    
    #pdb.set_trace()
    val_data = Dataset(train_dir, X_val, y_val, data_transforms=A_transforms, class_labels=class_labels, phase='val')
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    

    best_perf = 10000000000
    test_loss = None
    best_model = False
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )
    
    
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        num_epochs = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=-1
    )
    
    count = 0
    val_losses = []
    train_losses = []
    for epoch in range(begin_epoch, num_epochs):
        epoch_since = time.time()
        
        lr_scheduler.step()

        # train for one epoch
        train_loss = train(cfg, device, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, lamb=lamb)

        
        # evaluate on validation set
        perf_indicator = validate(
            cfg, device, val_loader, val_data, model, criterion,
            final_output_dir, tb_log_dir, writer_dict, lamb=lamb
        )

        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
            count = 0
            
        else:
            best_model = False
            count +=1
            
        
        
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
        

        val_losses.append(perf_indicator)
        train_losses.append(train_loss)
        if count == num_earlystop :
            break
        
        
        epoch_time_elapsed = time.time() - epoch_since
        print(f'epoch : {epoch}' \
                f' train loss : {round(train_loss,3)}' \
                              f' valid loss : {round(perf_indicator,3)}' \
                              f' Elapsed time: {int(epoch_time_elapsed // 60)}m {int(epoch_time_elapsed % 60)}s')
        

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

    time_elapsed = time.time() - since
    print('Training and Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation loss: {:4f}\n'.format(best_perf))

    if test_option == True :
        # test data
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True)
        
        model = initialize_model(model, cfg)
        parameters = f'{final_output_dir}/model_best.pth'
        
        model = model.to(device)
        model.load_state_dict(torch.load(parameters))
        
        test_loss = validate(
                cfg, device, test_loader, test_data, model, criterion,
                final_output_dir, tb_log_dir, writer_dict, lamb=lamb
            )
    
    print(f'test loss : {test_loss}')
    result_dict = {}
    result_dict['val_loss'] = val_losses
    result_dict['train_loss'] = train_losses
    result_dict['best_loss'] = best_perf
    result_dict['test_loss'] = test_loss
    result_dict['lr'] = lr
    with open(final_output_dir+'/result.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
    
if __name__ == '__main__':
    main()
