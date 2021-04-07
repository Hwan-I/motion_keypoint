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

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

import pdb

logger = logging.getLogger(__name__)

def train(config, device, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
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
    
    Returns
    -------
    losses.avg : float
        loss의 평균값 입니다.

    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        # input과 bbox 객체를 GPU에 넣을 수 있는 객체로 만듭니다.
        input = input.to(device)
        input = input.float()
        target = target.to(device)
        target = target.float()

        outputs = model(input)
        
        target = target.cuda(non_blocking=True)
        
        # target_weight를 반영합니다. 기본값은 0으로 되어있어 영향을 미치지 않습니다.
        target_weight = target_weight.cuda(non_blocking=True)
        
        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)
            
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

    return losses.avg

def validate(config, device, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
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

    Returns
    -------
    losses.avg : float
        예측된 heatmap loss의 평균값입니다.

    f_losses.avg : float
        예측된 keypoint loss의 평균값입니다.

    """
    
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    f_losses = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            
            # input과 bbox 객체를 GPU에 넣을 수 있는 객체로 만듭니다.
            input = input.to(device)
            input = input.float()
            target = target.to(device)
            target = target.float()
            
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs
            
            # 만약 TEST도 FLIP한다면 적용하는 옵션입니다.
            # 기본적으로는 False로 되어있어 통과합니다.
            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5
            
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # heatmap을 원래 keypoint 데이터로 만들기 위해 meta 데이터의 center, scale 값을 구합니다.
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            
            # 예측된 heatmap을 keypoint 데이터로 만듭니다.
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            

            criterion2 = torch.nn.MSELoss()
            
            trues = meta['origin'][:,:,:2]

            trues = trues.reshape(trues.shape[0],-1)
            
            # 예측된 keypoint 값을 실제 keypoint 값과 비교합니다.
            f_loss = criterion2(torch.from_numpy(preds.reshape(preds.shape[0],-1)), trues)
            f_losses.update(f_loss.item(), num_images)
            
            idx += num_images
            
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
        
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            
            writer_dict['valid_global_steps'] = global_steps + 1

    # 예측된 heatmap 값, keypoint 값을 반환합니다.
    return losses.avg, f_losses.avg


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
