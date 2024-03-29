# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import pdb

import json_tricks as json
import numpy as np

from dataset.JointsDataset import JointsDataset


import pickle

logger = logging.getLogger(__name__)


class Motion_dataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
        17: "neck",
        18: "left_palm",
        19: "right_palm",
        20: "spine2(back)",
        21: "spine1(waist)",
        22: "left_instep",
        23: "right_instep"
    },

    '''

    
    def __init__(self, cfg, root, image_set, is_train, phase, keypoints = None, transform=None):
        """
    
        각종 변수, dataset을 정의합니다.        
        
        Parameters
        ----------
        cfg : yacs.config.cfgNode
            config 파일
        root : str
            data 경로.
        image_set : like list
            image 파일 이름을 가진 list 객체.
        is_train : bool
            train 데이터인지 여부. train이면 True, valid나 test면 False.
        phase : str
            train인 데이터인 경우 'train', valid인 경우 'val', test인 경우 'test'
        keypoints : numpy.narray, optional
            keypoint 값을 가진 numpy 객체. The default is None.
        transform : torchvision.transforms, optional
            image에 transform을 넣는 객체. The default is None.
            
        Returns
        -------
        None.

        """
        super().__init__(cfg, root, is_train, transform)
        

        
        # 각종 파라미터를 정의합니다.
        self.image_set = image_set
        self.nms_thre = cfg.TEST.NMS_THRE
        self.is_train = is_train
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        
        self.phase = phase
        self.keypoints = keypoints
        self.transform = transform

        # load image file names
        self.image_set_index = [i for i in range(len(self.image_set))]
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))
        
        # keypoint 개수 정의
        self.num_joints = 24
        
        # 왼쪽, 오른쪽이 있는 신체 부위 class를 묶어서 pair로 정의합니다.
            # class 번호는 Motion_dataset 함수 바로 아래의 keypoints 부분을 참고하시기 바랍니다.
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16],
                           [18, 19], [22, 23]]
        self.parent_ids = None
        
        # 상체 부분 keypoint class 번호
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19,
                               20, 21)
        
        # 하체 부분 keypoint class 번호
        self.lower_body_ids = (11, 12, 13, 14, 15, 16, 22, 23)
        
        # image, keypoint, bbox를 가공하여 input data와 target 데이터를 정의합니다.
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))



    def _get_db(self):

        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            if self.phase == 'val' :
                gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        
        # annotation 데이터를 가져옵니다.
        with open(self.root + '/annotations/train_annotation.pkl', 'rb') as f:
            anns = pickle.load(f)
        
        # file 이름을 key로, file 이름에 해당하는 index 값을 value 값으로 하는 dict를 만듭니다.
        anns_name_dict = {}
        for i in range(len(anns)) :
            anns_name_dict[anns[i]['filename']] = i
            
        for i in range(len(self.image_set)):
            name = self.image_set[i]
            index = anns_name_dict[name]
            
            # annotation 값을 활용하여 image, target 값을 변형시킵니다.
            result = self._load_coco_keypoint_annotation_kernal(anns[index], i)
            if result != None :
                gt_db.append(result)

        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, obj, i):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        
        width = obj['width']
        height = obj['height']
        
        # sanitize bboxes
        x, y, w, h = obj['bbox']
        
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
            obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
        else :
            return None
        
        # target data를 3d로 만듭니다.
            # 원래는 z축에 사람이 잘 보이는지 여부를 넣어야 하지만 여기선 생략했습니다.
        joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
        joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
        for ipt in range(self.num_joints):
            joints_3d[ipt, 0] = obj['keypoints'][ipt * 2 + 0]
            joints_3d[ipt, 1] = obj['keypoints'][ipt * 2 + 1]
            joints_3d[ipt, 2] = 0
            
            t_vis = 1
            joints_3d_vis[ipt, 0] = t_vis
            joints_3d_vis[ipt, 1] = t_vis
            joints_3d_vis[ipt, 2] = 0
        
        image_path = f'/images/train_imgs/{self.image_set[i]}'
        
        # image의 center 값, scale 값을 추출합니다.
        center, scale = self._box2cs(obj['clean_bbox'][:4])
        
        rec = {
            'image': image_path,
            'center': center,
            'scale': scale,
            'joints_3d': joints_3d,
            'joints_3d_vis': joints_3d_vis,
            'filename': self.image_set[i],
            'imgnum': i,
        }

        return rec

    def _box2cs(self, box):
        """
        
        box를 입력받아 center, scale 값을 반환하는 함수로 연결시켜줍니다.

        Parameters
        ----------
        box : like list
            [x_min, y_min, width, height] 값을 가진 bbox 객체입니다.

        Returns
        -------
        TYPE
            center, scale 값을 반환하는 함수 값을 반환합니다.

        """
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        """
        image의 center, scale 값을 추출합니다.

        Parameters
        ----------
        x : float
            bbox의 x_min 값.
        y : float
            bbox의 y_min 값
        w : float
            bbox의 너비 값입니다.
        h : float
            bbox의 높이 값입니다.

        Returns
        -------
        center : float
            image의 중앙값입니다.
        scale : TYPE
            원래 image 크기 대비 bbox 크기에 따라 scaling 하는 값입니다.

        """
        
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale


    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results
