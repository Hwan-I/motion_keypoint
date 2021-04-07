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
from dataset.motion_dataset import Motion_dataset
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


# https://hoya012.github.io/blog/reproducible_pytorch/

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



def make_train_data(data_path):
    """
    dataset을 만들기 위한
    pandas의 데이터프레임에 있는 특정 epsg의 좌표값을 타겟 epsg 형태로 바꿔주는
    코드.
    
    Parameters
    ----------
    coor_list : numpy.ndarray
        위도,경도를 2차원형태로 가지고 있는 array.
        0번 column은 위도, 1번 column은 경도값을 가져야 함
        ex) np.array([[127.386131,36.37458], [127.392375,36.374389]])
        
        
    from_c : int
        해당 좌표의 epsg값. The default is 4326.
    to_c : TYPE, optional
        타겟(변환했을 때 결과) 좌표 epsg값. The default is 5179.
    point_option : TYPE, optional
        True면 변환한 좌표값을 Point 객체로 만들어 데이터프레임에 넣음. 
        기본값은 False.
        
    Returns
    -------
    s_pandas3_ : pandas.core.frame.DataFrame
        좌표를 바꾼뒤의 데이터프레임. lati는 위도, long은 경도.
        point_option이 True면 point라는 column은 좌표값을 geo.Point 객체로 만든 결과임.
    """
    
    train_df = pd.read_csv(data_path+'/train_df.csv')
    train_df = train_df.reset_index(drop=True)
    
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
    motions = train_df.iloc[:, 1:]
    columns = motions.columns.to_list()[::2]
    class_labels = [label.replace('_x', '').replace('_y', '') for label in columns]
    keypoints = []
    for motion in motions.to_numpy():
        a_keypoints = []
        for i in range(0, motion.shape[0], 2):
            a_keypoints.append((float(motion[i]), float(motion[i+1])))
        keypoints.append(a_keypoints)
    keypoints = np.array(keypoints)

    return imgs, keypoints, class_labels



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
    
    args = parse_args()
    update_config(cfg, args)
    
    lr = cfg.TRAIN.LR

    test_option = eval(cfg.test_option)
    RANDOM_SEED = int(cfg.RANDOMSEED)
    
    # https://hoya012.github.io/blog/reproducible_pytorch/
    
    np.random.seed(RANDOM_SEED) # cpu vars
    torch.manual_seed(RANDOM_SEED) # cpu  vars
    random.seed(RANDOM_SEED) # Python
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED) # Python hash buildin
    
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False
    
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, f'lr_{str(lr)}', 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    # cudnn related setting

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # annotations 생성
    if os.path.isfile(data_path+'/annotations/train_annotation.pkl') == False :
        make_annotations(data_path)


    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    
    shutil.copy2(
        os.path.join(this_dir, '../tools', 'train.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # define loss function (criterion) and optimizer


    criterion = JointsMSELoss(
        use_target_weight=False
    ).cuda()

    
    # Data Augumentation
    
    batch_size = int(cfg.TRAIN.BATCH_SIZE_PER_GPU)
    test_ratio = float(cfg.TEST_RATIO)
    
    imgs, keypoints, class_labels = make_train_data(data_path)
    
    num_epochs = cfg.TRAIN.END_EPOCH
    num_earlystop = num_epochs
    
    # trnasforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    since = time.time()
    
    if test_option == True :
        X_train, X_test, y_train, y_test = train_test_split(imgs, keypoints, test_size=0.1, random_state=RANDOM_SEED)
        test_dataset = [X_test, y_test]
        with open(final_output_dir+'/test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)
            
        test_data = Motion_dataset(cfg=cfg, root=data_path, image_set = X_test,
                         keypoints = y_test, is_train = False, phase='val',
                         transform= transforms.Compose([
                                 transforms.ToTensor(),                                 
                                 normalize
                                 ]))
        test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_ratio, random_state=RANDOM_SEED)
        
    else :
        X_train, X_val, y_train, y_val = train_test_split(imgs, keypoints, test_size=test_ratio, random_state=RANDOM_SEED)
    
 
    train_data = Motion_dataset(cfg=cfg, root=data_path, image_set = X_train,
                         keypoints = y_train, is_train = True, phase='train',
                         transform= transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                                 ]))
    
    val_data = Motion_dataset(cfg=cfg, root=data_path, image_set = X_val,
                         keypoints = y_val, is_train = False, phase='val',
                         transform= transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                                 ]))

    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    


    best_loss = 10000000000
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
        best_loss = checkpoint['perf']
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
    f_test_loss = None
    val_losses = []
    train_losses = []
    f_val_losses = []
    for epoch in range(begin_epoch, num_epochs):
        epoch_since = time.time()
        
        lr_scheduler.step()

        # train for one epoch
        train_loss = train(cfg, device, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
                            
        
        # evaluate on validation set
        val_loss, f_val_loss = validate(
            cfg, device, val_loader, val_data, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if f_val_loss <= best_loss:
            best_loss = f_val_loss
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
            'perf': best_loss,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
        

        val_losses.append(val_loss)
        train_losses.append(train_loss)
        f_val_losses.append(f_val_loss)
        if count == num_earlystop :
            break
        
        
        epoch_time_elapsed = time.time() - epoch_since
        print(f'epoch : {epoch}' \
                f' train loss : {round(train_loss,3)}' \
                    f' val loss : {round(val_loss,3)}' \
                              f' true val loss : {round(f_val_loss,3)}' \
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
    print('Best validation loss: {:4f}\n'.format(best_loss))
    
    if test_option == True :
        # test data
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )
        
        parameters = f'{final_output_dir}/model_best.pth'
        
        model = model.to(device)
        model.load_state_dict(torch.load(parameters))
        
        test_loss, f_test_loss = validate(
                cfg, device, test_loader, test_data, model, criterion,
                final_output_dir, tb_log_dir, writer_dict
            )
    
    print(f'test loss : {f_test_loss}')
    result_dict = {}
    result_dict['val_loss'] = val_losses
    result_dict['train_loss'] = train_losses
    result_dict['best_loss'] = best_loss
    result_dict['test_loss'] = f_test_loss
    result_dict['f_val_loss'] = f_val_losses
    result_dict['lr'] = lr
    with open(final_output_dir+'/result.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    

if __name__ == '__main__':
    main()
