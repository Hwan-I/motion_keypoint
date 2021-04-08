#### HRNet을 활용하여 bbox를 추정하는 모델입니다.
* 최종 제출에 사용한 config 파일 : w48_384x288_adam_lr1e-3_03_origin.yaml
* train 및 test 결과는 output 폴더 안에서 생성된 

#### 앞의 Readme.md의 환경 셋팅을 마친 뒤 train과 test는 다음과 같이 하면 됩니다.(앞의 Readme.md의 방법대로 하셔도 됩니다)
* --cfg : config 파일을 의미합니다. config 파일에서 각종 파라미터를 수정할 수 있습니다. 
  * 폴더 안에 있는 다른 config 파일을 쓰셔도 됩니다. 
* --test_option : 데이터를 train, valid, test로 나누는 옵션입니다. 
  * True : 10%를 test 데이터로 따로 빼며 나머지 데이터에서 config 파일에 명시된 TEST_RATIO 변수를 기준으로 train과 valid를 나눕니다.
  * False : test 데이터 없이 config 파일에 명시된 TEST_RATIO 변수를 기준으로 train과 valid를 나눕니다.
* 아래 예시는 Private Score를 만드는데 사용한 코드입니다.

```
python tools/train.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_03_origin.yaml --test_option False
```
* 만약 cuda 버전이 10.2 이상이며 CUBLAS_WORKSPACE_CONFIG=:16:8를 추가하고자 한다면 아래와 같이 명령어를 쓰시면 됩니다.
```
CUBLAS_WORKSPACE_CONFIG=:16:8 python tools/train.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_03_origin.yaml --test_option False
```

#### Test : test_imgs에 대해 실행시키는 것으로 아래의 코드를 실행시키면 됩니다.
* --cfg : config 파일을 의미합니다.
* --output_path : model_best.pth 등 파라미터 값과 결과가 저장된 폴더 위치를 의미합니다. output 이후의 경로부터 쓰시면 됩니다.
* test 결과는 output_path로 나옵니다. (파일 이름 : test_annotation.pkl)
* 아래 예시는 Private Score를 만드는데 사용한 코드입니다.

```
python tools/test.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_03_origin.yaml --output_path output/lr_0.001/coco/pose_hrnet/w48_384x288_adam_lr1e-3_03_origin
```
