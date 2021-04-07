## 2021년 모션 키포인트 검출 AI 경진대회 코드입니다. (닉네임 : 환이, Private : 7.511)

* 전체적인 코드는 [HRNet code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch#readme)를 참조하여 만들었습니다.
* bbox, keypoint를 만드는 코드가 폴더로 나뉘어 있습니다.
  * 둘이 거의 유사한 코드라 용도에 따라 2개를 합칠 수도 있습니다.

### 컴퓨터 환경
* CPU : AMD Ryzen 7 3700X
* GPU : RTX 3080 (1개)
* RAM : 32GB (Swap : 64GB)
* 운영체제 : Ubuntu 18.04.5 LTS

### 학습 소요 시간
* bbox : 10시간 (1 epoch 당 평균 3분 소요)
* keypoint : 12시간 (1 epoch 당 평균 3분 40초 소요)

### 환경 셋팅
※ 괄호로 표현한 부분은 임의의 변수명으로 넣으시면 됩니다.

1. 가상환경 셋팅(anaconda 3 기준)
```
conda create -n (환경 이름) python=3.6
```

2. 패키지 설치 : HRNet_bbox나 HRNet_keypoint 폴더로 이동하여 아래의 명령어를 실행시켜주시면 됩니다. (한 곳에서만 실행시키면 양쪽 모두 사용 가능합니다)
* cuda의 경우 cu111로 되어 있는데 이는 RTX3080 기준입니다. torch 버전만 맞추시고 cuda는 각 그래픽 카드에 맞는 버젼을 설치하시기 바랍니다.
  * [이 사이트](https://pytorch.org/get-started/previous-versions/)를 참조하시기 바랍니다.

```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -U git+https://github.com/albumentations-team/albumentations
```

3. 가중치 파일 다운로드 : bbox, keypoint 둘 다 같으며, 아래에서 사용할 가중치 파일을 [Google](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC) 또는 [OneDrive](https://onedrive.live.com/?cid=56b9f9c97f261712&id=56B9F9C97F261712%2111773&authkey=%21AEwfaSueYurmSRA)에서 다운받아 아래의 폴더에 이동시키시면 됩니다.

  * bbox 모델에 설치시, POSE_ROOT는 HRNet_bbox이고, keypoint 모델에 설치시, POSE_ROOT는 HRNet_keypoint입니다.
  * 아래와 같은 폴더 형태로 만들어야 합니다.
  * 실험에서는 pose_coco의 pose_hrnet_w48_384x288.pth가 성능이 제일 좋아 이를 사용했습니다.
```
${POSE_ROOT}
 `-- models
     `-- pytorch
         |-- imagenet
         |   |-- hrnet_w32-36af842e.pth
         |   |-- hrnet_w48-8ef0771d.pth
         |   |-- resnet50-19c8e357.pth
         |   |-- resnet101-5d3b4d8f.pth
         |   `-- resnet152-b121ed2d.pth
         |-- pose_coco
         |   |-- pose_hrnet_w32_256x192.pth
         |   |-- pose_hrnet_w32_384x288.pth
         |   |-- pose_hrnet_w48_256x192.pth
         |   |-- pose_hrnet_w48_384x288.pth
         |   |-- pose_resnet_101_256x192.pth
         |   |-- pose_resnet_101_384x288.pth
         |   |-- pose_resnet_152_256x192.pth
         |   |-- pose_resnet_152_384x288.pth
         |   |-- pose_resnet_50_256x192.pth
         |   `-- pose_resnet_50_384x288.pth
         `-- pose_mpii
             |-- pose_hrnet_w32_256x256.pth
             |-- pose_hrnet_w48_256x256.pth
             |-- pose_resnet_101_256x256.pth
             |-- pose_resnet_152_256x256.pth
             `-- pose_resnet_50_256x256.pth
```
4. data 파일
* data 파일 구조는 아래와 같은 구조여야 합니다. (bbox, keypoint 모두 해당)
* 확장자 명이 없으면 폴더입니다.
* keypoint의 경우 annotations 폴더 안에 test_annotation.pkl 파일이 필요합니다.
  * bbox model에서 test image에 대해 annotation을 생성해서 만든 폴더를 annotations 폴더 안에 넣으시면 됩니다. 
```
${POSE_ROOT}
|-- data
`-- |-- train_df.csv
    |-- sample_submission.csv
    |-- annotations
    |-- images
        |-- train_imgs
            |-- 001-1-1-01-Z17_A-0000001.jpg
            |-- 001-1-1-01-Z17_A-0000003.jpg
            |-- ...
        |-- test_imgs
            |-- 649-2-4-32-Z148_A-0000001.jpg
            |-- 649-2-4-32-Z148_A-0000003.jpg
            |-- ...
```

### Train and Test 
* 이 부분은 HRNet_keypoint, HRNet_bbox에서 사용하는 config 파일이 다르기 때문에 Private Score을 유사하게 재현하고자 한다면 해당 repository의 Readme.md를 참조하시기 바랍니다

#### Train : 아래의 코드를 실행시키면 됩니다.
* --cfg : config 파일을 의미합니다. config 파일에서 각종 파라미터를 수정할 수 있습니다.
  * config 파일 위치 : ${POSE_ROOT}/experiments (coco dataset를 쓴다면 ${POSE_ROOT}/experiments/coco/hrnet의 config 파일을 쓰시면 됩니다)
* --test_option : 데이터를 train, valid, test로 나누는 옵션입니다. 
  * True : 10%를 test 데이터로 따로 빼며 나머지 데이터에서 config 파일에 명시된 TEST_RATIO 변수를 기준으로 train과 valid를 나눕니다.
  * False : test 데이터 없이 config 파일에 명시된 TEST_RATIO 변수를 기준으로 train과 valid를 나눕니다.
* CUBLAS_WORKSPACE_CONFIG=:16:8 : CUDA version이 10.2 이상인 경우 gpu의 랜덤을 잡아주기 위해 추가한 값입니다.([참조사이트](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility))
  * 저 부분 빼고 python tools/train.py ... 으로 실행하셔도 됩니다.
예를 들어 coco dataset의 384x288 size의 input feature map을 사용하고 싶다면 아래와 같이 쓰면 됩니다.
```
CUBLAS_WORKSPACE_CONFIG=:16:8 python tools/train.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_03_origin.yaml --test_option False
```

#### Test : test_imgs에 대해 실행시키는 것으로 아래의 코드를 실행시키면 됩니다.
* --cfg : config 파일을 의미합니다.
* --output_path : model_best.pth 등 파라미터 값과 결과가 저장된 폴더 위치를 의미합니다. output 이후의 경로부터 쓰시면 됩니다.
* test 결과는 output_path로 나옵니다.
```
python tools/test.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml --output_path output/lr_0.001/coco/pose_hrnet/w48_384x288_adam_lr1e-3_03_origin
```
