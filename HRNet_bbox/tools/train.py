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



class Mynet(nn.Module):
    """
    
    끝부분에 bbox를 만들기 위해 붙이는 fully connected layer입니다.
    
    """
    def __init__(self, cfg):
        super(Mynet, self).__init__()
        
        num_classes = cfg['MODEL']['NUM_JOINTS']
        BN_MOMENTUM = 0.1

        
        if cfg['MODEL']['IMAGE_SIZE'][1] == 384 :
            if cfg['MODEL']['IMAGE_SIZE'][1] == cfg['MODEL']['IMAGE_SIZE'][0] :
                self.bbox_layer = nn.Linear(num_classes*72*72, 4)

            else :
                self.bbox_layer = nn.Linear(num_classes*72*96, 4)

        elif cfg['MODEL']['IMAGE_SIZE'][1] == 256 :
            self.bbox_layer = nn.Linear(num_classes*64*64, 4)

        else :
            self.bbox_layer = nn.Linear(num_classes*48*48, 4)

        
        self.bn1 = nn.BatchNorm2d(num_classes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)
        
        
    def forward(self, x):

        x = x.view(x.shape[0],-1)
        
        b_y = self.bbox_layer(x)

        return b_y
        


# https://github.com/amdegroot/pytorch-containers/blob/master/README.md
def initialize_model(model_ft, cfg):
    """
    모델의 끝 부분에 layer를 추가하고 channel 개수를 바꾸어 bbox 출력 모델로 만듭니다.

    Parameters
    ----------
    model_ft : model
        모델 객체.
    cfg : yacs.config.cfgNode
        config 파일

    Returns
    -------
    f_model : model
        bbox로 변형시킨 모델. 

    """
    
    # 뒷 부분에 추가할 부분
    mynet = Mynet(cfg)
    
    # 뒷부분에 추가되는 레이어의 가중치를 xavier uniform으로 초기화 합니다.
    torch.nn.init.xavier_uniform_(mynet.bbox_layer.weight)
    
    # 뒷부분을 붙입니다.
    f_model = nn.Sequential(
            model_ft,
            mynet
            )
    
    
    return f_model




def make_train_data(data_path):
    """
    
    bbox를 만드는 train 데이터를 만듭니다.
    ※ HRNet에서 bbox를 쓸 때 기본 형태는 
    [왼쪽 꼭지점 x좌표, 아래 꼭지점 y좌표, 너비, 높이]입니다. 그렇기 때문에 해당 
    형태로 bbox 값을 변경합니다.
    
    ※ delete list에서 쓰지 않을 train image의 번호를 넣을 수 있습니다.
    
    Parameters
    ----------
    data_path : str
        data의 path입니다. data 폴더 위치를 나타냅니다.

    Returns
    -------
    imgs : numpy.ndarry
        이미지 파일 이름을 numpy 객체에 담은 형태입니다.
    bbox : list
        각 image 파일의 bbox값을 담은 list 객체입니다.
    class_labels : list
        각 image 파일이 bbox에 가진 객체가 무엇인지 나타낸 list입니다.

    """
    
    # train 이미지 정보를 가진 dataframe을 불러옵니다.
    train_df = pd.read_csv(data_path+'/train_df.csv')
    train_df = train_df.reset_index(drop=True)
    
    # 만들었던 annotation 파일을 불러옵니다.
    with open(data_path+'/annotations/train_annotation.pkl', 'rb') as f:
        annotations = pickle.load(f)
    
    anno_df = pd.DataFrame()
    filenames = []
    
    # bbox를 coco dataset의 형태로 맞춰줍니다.
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

    
    train_df = train_df.iloc[:, 0]
    train_df = pd.merge(train_df, anno_df, on='image', how='left')
    
    # 쓰지 않을 train 파일의 index 번호입니다.
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
    
    # img 파일이름을 추출합니다.
    imgs = train_df.iloc[:, 0].to_numpy()
    
    # 각 img 파일에 대응하는 bbox 객체를 추출합니다.
    bbox = train_df.iloc[:, 1:].values.tolist()
    
    # bbox의 class는 사람 1명만 있기 때문에 'person' 1개만 넣습니다.
    class_labels = ['person']
    for i, box in enumerate(bbox) :
        bbox[i] = [box]

    return imgs, bbox, class_labels


def make_annotations(data_path) :
    """
    train image의 bbox를 만들기 위한 meta정보를 가진 annotation 파일을 만듭니다.

    Parameters
    ----------
    data_path : str
        data의 path입니다. data 폴더 위치를 나타냅니다.

    Returns
    -------
    None.

    """
    
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
    
    # 주요 path 정의
    data_path = './data'
    train_dir = Path(data_path, 'images/train_imgs')
    
    # config 파일을 가져옵니다.
    args = parse_args()
    update_config(cfg, args)

    lr = cfg.TRAIN.LR
    lamb = cfg.LAMB
    test_option = eval(cfg.test_option)
    
    input_w = cfg.MODEL.IMAGE_SIZE[1]
    input_h = cfg.MODEL.IMAGE_SIZE[0]
    
    # 랜덤 요소를 최대한 줄여줌
    RANDOM_SEED = int(cfg.RANDOMSEED)
    np.random.seed(RANDOM_SEED) # cpu vars
    torch.manual_seed(RANDOM_SEED) # cpu  vars
    random.seed(RANDOM_SEED) # Python
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED) # Python hash buildin
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU

    
    # log 데이터와 최종 저장위치를 만듭니다.
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, f'lr_{str(lr)}', 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # annotation 파일을 만듭니다.
    if os.path.isfile(data_path+'/annotations/train_annotation.pkl') == False :
        make_annotations(data_path)
    
    # 쓰려는 모델을 불러옵니다.
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    
    # model의 끝부분 수정 및 초기화 작업을 진행합니다.
    model = initialize_model(model, cfg)
    
    
    # model 파일과 train.py 파일을 copy합니다.
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
    
    
    # model을 그래픽카드가 있을 경우 cuda device로 전환합니다.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # loss를 정의합니다.
    criterion = nn.MSELoss().cuda()

    # Data Augumentation을 정의합니다.
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
    

    # parameter를 설정합니다.
    batch_size = int(cfg.TRAIN.BATCH_SIZE_PER_GPU)
    test_ratio = float(cfg.TEST_RATIO)
    num_epochs = cfg.TRAIN.END_EPOCH
    
    # earlystopping에 주는 숫자 변수입니다.
    num_earlystop = num_epochs
    
    # torch에서 사용할 dataset을 생성합니다.
    imgs, bbox, class_labels = make_train_data(data_path)

    since = time.time()
    
    """
    # test_option : train, valid로 데이터를 나눌 때 test data를 고려할지 결정합니다.
        * True일 경우 test file을 10% 뺍니다.
        * False일 경우 test file 빼지 않습니다.
    """
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
    
    val_data = Dataset(train_dir, X_val, y_val, data_transforms=A_transforms, class_labels=class_labels, phase='val')
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    
    # best loss를 판별하기 위한 변수 초기화
    best_perf = 10000000000
    test_loss = None
    best_model = False
    
    # optimizer 정의
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )
    
    # 중간에 학습된 모델이 있다면 해당 epoch에서부터 진행할 수 있도록 만듭니다.
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
    
    # lr_scheduler 정의
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=-1
    )
    
    # early stopping하는데 사용하는 count 변수
    count = 0
    val_losses = []
    train_losses = []
    
    # 학습 시작
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
        
        # 해당 epoch이 best_model인지 판별합니다. valid 값을 기준으로 결정됩니다.
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
        
        # loss를 저장합니다.
        val_losses.append(perf_indicator)
        train_losses.append(train_loss)
        if count == num_earlystop :
            break
        
        
        epoch_time_elapsed = time.time() - epoch_since
        print(f'epoch : {epoch}' \
                f' train loss : {round(train_loss,3)}' \
                              f' valid loss : {round(perf_indicator,3)}' \
                              f' Elapsed time: {int(epoch_time_elapsed // 60)}m {int(epoch_time_elapsed % 60)}s')
        
    # log 파일 등을 저장합니다.
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
    
    # test_option이 True일 경우, 떼어난 10% 데이터에 대해 만들어진 모델로 eval을 진행합니다.
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
    
    # loss 결과를 pickle 파일로 따로 저장합니다.
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
