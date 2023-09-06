# CLIP_CAP_Based
CLIPCap 논문을 Base로 구현한 AI 코드

# File Structure
```
|-- ./CLIP_CAP_Based
|   |-- ./CLIP_CAP_Based/data
|   |-- ./CLIP_CAP_Based/EDA
|   |-- ./CLIP_CAP_Based/Inference_notebook
|   |-- ./CLIP_CAP_Based/parse_caption.ipynb
|   |-- ./CLIP_CAP_Based/sample_images
|   `-- ./CLIP_CAP_Based/train_custom.py
|   |-- ./CLIP_CAP_Based/README.md
|-- ./data
|   |-- ./data/sample_submission.csv
|   |-- ./data/train.csv
|   |-- ./data/test.csv
|   |-- ./data/submit.csv
|   |-- ./data/train
|   |-- ./data/test
|   |-- ./data/train_data.pkl
|   `-- ./data/train_data_tokens.pkl -> Made by parse_caption.ipynb
```
# 실행 순서
1. 경로를 위의 File Structure대로 구성한다.
2. `CLIP_CAP_Based/parse_caption.ipynb`를 이용해 학습에 필요한 train_data, token들을 pickle 파일로 변환한다.
    - ResNet-based CLIP으로 Feature Extracting 방식
      ```
      python parse_coco.py --clip_model_type RN50x4
      ```
    - VIT-based CLIP으로 Feature Extracting 방식  -> **추천**
      ```
      python parse_coco.py --clip_model_type ViT-B/32
      ```
3. `CLIP_CAP_Based/train_custom.py`를 이용해 학습을 시작함.
    - fine-tuning GPT2 방식으로 실행하는 코드 -> **GPU VRAM 크기가 낮으면 추천**
      ```
      python train_custom.py --data ../data/train_data.pkl --out_dir ./custom_coco_train/ --epochs 11
      ``` 
    - transformer mapping network를 이용하는 코드 -> **GPU VRAM 크기가 높으면 추천**
      ```
      python train_custom.py --only_prefix --data ../data/train_data.pkl --out_dir ./custom_coco_train/ --mapping_type transformer  --prefix_length 40 --prefix_length_clip 40
      ```
4. Inference_notebook을 이용해 trained-model과 Beam Search Inference 방식을 이용해 결과 시각화 해보기.
5. Submit.ipynb를 이용해 대회 submit 코드 만들기 - **추가 예정**
6. VIT Feature를 가져와서 MOS Regression Task도 가능하도록 parse_caption에서부터 코드 수정하기 - **추가 예정**

# Environments
- pytorch:1.13.0
- cudnn8.0
- cuda11.6

# Reference
- https://github.com/rmokady/CLIP_prefix_caption
- https://arxiv.org/pdf/2111.09734.pdf
