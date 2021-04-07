# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import pdb

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
#from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, device, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, lamb=1):
    """
    1 epoch train 시킵니다.

    Parameters
    ----------
    config : yacs.config.CfgNode
        config 파일입니다.
    device : torch.device
        GPU 사용시 데이터를 GPU에 넣어주는 객체입니다.
    train_loader : torch.utils.data.dataloader.DataLoader
        train data Loader.
    model : model
        학습하는 모델 객체입니다.
    criterion : torch.nn.modules.loss
        torch의 loss 객체입니다.
    optimizer : torch.optim
        torch의 optimizer 객체입니다.
    epoch : int
        현재 epoch 값입니다.
    output_dir : str
        결과값이 저장될 경로입니다.
    tb_log_dir : str
        log 파일 위치입니다.
    writer_dict : dict
        실험 기록 dict입니다.
    lamb : int, optional
        lamb값으로 loss값에 쓰이는 값입니다. The default is 1.
        (현재는 쓰이지 않음)

    Returns
    -------
    losses.avg : float
        loss의 평균값 입니다.

    """
    
    # 각종 평균값을 기록하는 변수
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    for i, (input, bboxes) in enumerate(train_loader):
        
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        # input과 bbox 객체를 GPU에 넣을 수 있는 객체로 만듭니다.
        input = input.to(device)
        bboxes = bboxes.to(device)

        b_outputs = model(input)
        b_outputs = b_outputs.cuda(non_blocking=True)
        
        # output 값을 원래 사이즈(1920 * 1080)로 만듭니다.
        b_outputs[:, 0] /= config.INPUT_SIZE / 1920
        b_outputs[:, 1] /= config.INPUT_SIZE / 1080
        b_outputs[:, 2] /= config.INPUT_SIZE / 1920
        b_outputs[:, 3] /= config.INPUT_SIZE / 1080
        
        bboxes[:, 0] /= config.INPUT_SIZE / 1920
        bboxes[:, 1] /= config.INPUT_SIZE / 1080
        bboxes[:, 2] /= config.INPUT_SIZE / 1920
        bboxes[:, 3] /= config.INPUT_SIZE / 1080
        
        # 원래 사이즈로 만들고 loss를 게산합니다.
        box_loss = criterion(b_outputs.float(), bboxes.float())
        
        loss = box_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
           
            writer_dict['train_global_steps'] = global_steps + 1
            


    return losses.avg
    
    
def validate(config, device, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, lamb=1):
    """
    valid data를 모델에 넣어 모델을 평가합니다.

    Parameters
    ----------
    config : yacs.config.CfgNode
        config 파일입니다.
    device : torch.device
        GPU 사용시 데이터를 GPU에 넣어주는 객체입니다.
    val_loader : torch.utils.data.dataloader.DataLoader
        validation data Loader.
    val_dataset : dataset.dataset
        validation dataset.
    model : model
        학습하는 모델 객체입니다.
    criterion : torch.nn.modules.loss
        torch의 loss 객체입니다.
    output_dir : str
        결과값이 저장될 경로입니다.
    tb_log_dir : str
        log 파일 위치입니다.
    writer_dict : dict, optional
        실험 기록 dict입니다. The default is None.
    lamb : int, optional
        lamb값으로 loss값에 쓰이는 값입니다. The default is 1.
        (현재는 쓰이지 않음)

    Returns
    -------
    losses.avg : float
        loss의 평균값 입니다.

    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, bboxes) in enumerate(val_loader):
            
            # input과 bbox 객체를 GPU에 넣을 수 있는 객체로 만듭니다.
            input = input.to(device)
            bboxes = bboxes.to(device)
            
            # compute output
            b_outputs = model(input)
            b_outputs = b_outputs.cuda(non_blocking=True)

            # output 값을 원래 사이즈(1920 * 1080)로 만듭니다.
            b_outputs[:, 0] /= config.INPUT_SIZE / 1920
            b_outputs[:, 1] /= config.INPUT_SIZE / 1080
            b_outputs[:, 2] /= config.INPUT_SIZE / 1920
            b_outputs[:, 3] /= config.INPUT_SIZE / 1080
            
            bboxes[:, 0] /= config.INPUT_SIZE / 1920
            bboxes[:, 1] /= config.INPUT_SIZE / 1080
            bboxes[:, 2] /= config.INPUT_SIZE / 1920
            bboxes[:, 3] /= config.INPUT_SIZE / 1080

            # 원래 사이즈로 만들고 loss를 게산합니다.
            box_loss = criterion(b_outputs.float(), bboxes.float())
            
            loss = box_loss
            
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            
            writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
