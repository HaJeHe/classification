import pandas as pd
import numpy as np
from tqdm import tqdm   #진행률 프로세스바
from glob import glob   #사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
from typing import *  # a: int = 3
from IPython.display import Image as IPImage  #Image  출력
from sklearn.model_selection import train_test_split
import random
import os
import shutil
import sys
from PIL import Image as Image  # Opencv 와 같은 이미지 처리 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim #PyTorch에서 제공하는 최적화(optimizer) 알고리즘을 포함하는 모듈 adam
from torchvision import datasets, models, transforms
"""  torchvision  주요기능
1. 데이터셋 접근 :  MNIST, Imagenet 등 사전 정의된 데이터셋을 제공
2. 데이터 변환 : 이미지 데이터를 전처리하거나 증강 (이미지 크기 조정, 회전, 뒤집기)
3. 모델 : ResNet, AlexNet등 유명한 모델을 쉽게 불러와 사용 """
from torch.utils.data import DataLoader, Dataset #Dataset 은 샘플과 정답을 저장,  DataLoader는 
# Dataset  을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체로 감쌈
from torch.utils.data import ConcatDataset
#confusion matrix 형태의 데이터관리
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score  
import seaborn as sns #데이터 시각화
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import subprocess
import pprint 


print('gpu_test.py')



# 사전 학습된 ResNet18 모델 불러오기
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

# 모델의 마지막 완전 연결 계층(fc)을 사용자 정의 계층으로 대체
# 이 계층은 in_features에서 10개의 출력으로 매핑합니다 (10개 클래스 분류를 위함)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 5),
)

# 교차 엔트로피 손실 함수 초기화
criterion = nn.CrossEntropyLoss()
# 최적화 알고리즘으로 Adam 사용
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


model.load_state_dict(torch.load('best_model_resnet18.pth', weights_only = True))
model.eval()

image_path = "d:/python_code/resnet18/dataset/digit_data"
asset_path = "d:/python_code/resnet18/assets"

# 학습 데이터셋
origin_train_df = pd.read_csv(f"{image_path}/train_data.txt", names=["path"])
origin_train_df["label"] = origin_train_df["path"].str[0].astype(int)
origin_train_df["path"] = image_path + "/" + origin_train_df["path"]
# print(origin_train_df)

# 테스트 데이터셋 (valid_data.txt지만 미리 나눠진 test set으로 사용하도록 하겠음)
test_df = pd.read_csv(f"{image_path}/valid_data.txt", names=["path"])
test_df["label"] = test_df["path"].str[0].astype(int)
test_df["path"] = image_path + "/" + test_df["path"]
# print(test_df)

train_df, valid_df = train_test_split(
    origin_train_df, # 분할시킬 데이터 입력
    test_size=0.2,   # 테스트 데이터셋의 비율
    random_state=0,  # 셧플의 시드값
    shuffle=True,
    stratify=origin_train_df["label"], 
    
)

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        # 데이터셋 초기화
        # dataframe: 이미지 경로와 레이블이 포함된 데이터프레임
        # transform: 이미지에 적용할 전처리(transform) 함수
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        # 데이터셋의 총 샘플 수 반환
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 주어진 인덱스(idx)에 해당하는 샘플을 반환
        # 이미지 경로를 데이터프레임에서 가져옴
        img_name = self.dataframe.iloc[idx, 0]
        # 이미지 파일을 열고 RGB 모드로 변환
        img = Image.open(img_name).convert('RGB')
        # 레이블 정보를 정수형으로 가져옴
        label = int(self.dataframe.iloc[idx, 1])

        # transform이 지정되어 있다면 이미지에 전처리를 적용
        if self.transform:
            img = self.transform(img)

        # 이미지와 레이블을 반환
        return img, label
    
    
channel_means = [107.0054197492284, 101.17760866898148, 84.9624026234568]
channel_stds = [29.888622375757215, 25.865585869030646, 20.95805035580827]
print("Overall Data: Mean -", channel_means, "Std -", channel_stds)

# 채널 평균 및 표준편차를 0~1 사이의 값으로 정규화
normalized_channel_means = [x / 255 for x in channel_means]
normalized_channel_stds = [x / 255 for x in channel_stds]

# 이미지 전처리를 위한 변환 작업 정의.
# transforms.Normalize 이전에 이미 픽셀값은 0~1 사이로 정규화 되어 있습니다.
# 따라서 기존의 0~255 픽셀값 기준에서 도출된 channel_means, channel_stds 도 0~1 사이 값으로 정규화 된 normalized_channel_means, normalized_channel_stds 로 변환하여 사용합니다
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalized_channel_means, std=normalized_channel_stds),
])


if __name__ == "__main__":
    batch_size = 16
    train_dataset = CustomDataset(dataframe=train_df, transform=transform)
    valid_dataset = CustomDataset(dataframe=valid_df, transform=transform)
    test_dataset = CustomDataset(dataframe=test_df, transform=transform)
    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    conf_mat = confusion_matrix(all_labels, all_predictions)
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_mat_normalized, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('resnet18 Confusion Matrix')
    plt.show()