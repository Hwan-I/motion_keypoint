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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #acc = AverageMeter()
    
    # switch to train mode
    model.train()
    
    end = time.time()
    
    for i, (input, bboxes) in enumerate(train_loader):
        
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = input.to(device)
        bboxes = bboxes.to(device)

        # compute output
        
        #b_outputs, k_outputs = model(input)
        b_outputs = model(input)

        b_outputs = b_outputs.cuda(non_blocking=True)
        #k_outputs = k_outputs.cuda(non_blocking=True)

        
        b_outputs[:, 0] /= config.INPUT_SIZE / 1920
        b_outputs[:, 1] /= config.INPUT_SIZE / 1080
        b_outputs[:, 2] /= config.INPUT_SIZE / 1920
        b_outputs[:, 3] /= config.INPUT_SIZE / 1080
        
        bboxes[:, 0] /= config.INPUT_SIZE / 1920
        bboxes[:, 1] /= config.INPUT_SIZE / 1080
        bboxes[:, 2] /= config.INPUT_SIZE / 1920
        bboxes[:, 3] /= config.INPUT_SIZE / 1080
        
        #key_loss = criterion(k_outputs.float(), target.float())
        box_loss = criterion(b_outputs.float(), bboxes.float())
        
        #loss = lamb * key_loss + box_loss
        loss = box_loss
        # loss = criterion(output, target, target_weight)

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
            #writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            
            """
            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            
            save_debug_images(config, input, target, pred*4, output,
                              prefix)
            """

    return losses.avg
    
    
def validate(config, device, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, lamb=1):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, bboxes) in enumerate(val_loader):
            
            input = input.to(device)
            bboxes = bboxes.to(device)
            
            # compute output
            #b_outputs, k_outputs = model(input)
            b_outputs = model(input)

            b_outputs = b_outputs.cuda(non_blocking=True)
            #k_outputs = k_outputs.cuda(non_blocking=True)
            #target_weight = target_weight.cuda(non_blocking=True)

            b_outputs[:, 0] /= config.INPUT_SIZE / 1920
            b_outputs[:, 1] /= config.INPUT_SIZE / 1080
            b_outputs[:, 2] /= config.INPUT_SIZE / 1920
            b_outputs[:, 3] /= config.INPUT_SIZE / 1080
            
            bboxes[:, 0] /= config.INPUT_SIZE / 1920
            bboxes[:, 1] /= config.INPUT_SIZE / 1080
            bboxes[:, 2] /= config.INPUT_SIZE / 1920
            bboxes[:, 3] /= config.INPUT_SIZE / 1080

            #key_loss = criterion(k_outputs.float(), target.float())
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
