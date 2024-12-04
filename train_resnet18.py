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
import numpy as np
from torchvision.transforms import functional as F
import pprint 
from PreP import get_gpu_info, show_bar_01, compute_overall_mean_std 
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())


pprint.pprint(get_gpu_info())

exit()


def seed_everything(seed):
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch cuda
    torch.backends.cudnn.deterministic = True  # CuDNN
    torch.backends.cudnn.benchmark = False  # CuDNN benchmark
seed_everything(seed=32)
# print(os.getcwd())

image_path = "d:/python_code/resnet18/dataset/digit_data"
asset_path = "d:/python_code/resnet18/assets"


print(f"{image_path}/train_data.txt" )


# 학습 데이터셋
origin_train_df = pd.read_csv(f"{image_path}/train_data.txt", names=["path"])
origin_train_df["label"] = origin_train_df["path"].str[0].astype(int)
origin_train_df["path"] = image_path + "/" + origin_train_df["path"]
# print(origin_train_df)
# print(origin_train_df.columns,end='\n')
# print(origin_train_df.columns[1:])



# 테스트 데이터셋 (valid_data.txt지만 미리 나눠진 test set으로 사용하도록 하겠음)
test_df = pd.read_csv(f"{image_path}/valid_data.txt", names=["path"])
test_df["label"] = test_df["path"].str[0].astype(int)
test_df["path"] = image_path + "/" + test_df["path"]
# print(test_df)


# show_bar_01(origin_train_df,test_df)
# exit()

train_df, valid_df = train_test_split(
    origin_train_df, # 분할시킬 데이터 입력
    test_size=0.2,   # 테스트 데이터셋의 비율
    random_state=0,  # 셧플의 시드값
    shuffle=True,
    stratify=origin_train_df["label"], 
    
)
'''stratify : 지정한 Data의 비율을 유지한다. 예를 들어, Label Set인 Y가 25%의 0과 75%의 1로 이루어진 
    Binary Set일 때, stratify=Y로 설정하면 나누어진 데이터셋들도 0과 1을 각각 25%, 75%로 유지한 채 분할된다.'''
# print('#'*100)
# print(train_df)
# print('#'*100)
# print(valid_df)

       

# train_df와 valid_df에 대해서 계산
# 컴퓨팅 자원이 풍부한 환경에서 실행할 경우 직접 실행해볼 수 있습니다.
# channel_means, channel_stds = compute_overall_mean_std([train_df, valid_df])
# print("Overall Data: Mean -", channel_means, "Std -", channel_stds)
channel_means = [107.0054197492284, 101.17760866898148, 84.9624026234568]
channel_stds = [29.888622375757215, 25.865585869030646, 20.95805035580827]

# 채널 평균 및 표준편차를 0~1 사이의 값으로 정규화
normalized_channel_means = [x / 255 for x in channel_means]
normalized_channel_stds = [x / 255 for x in channel_stds]

# 데이터 전처리(transform) 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지를 224x224로 리사이즈
    transforms.ToTensor(),  # 이미지를 텐서로 변환 (0-255 값을 0-1 범위로 변환)
    transforms.Normalize(mean=normalized_channel_means, std=normalized_channel_stds),  # 채널별 평균과 표준편차로 정규화
])

# 데이터 증강을 위한 transform 정의
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지를 224x224로 리사이즈
    transforms.RandomRotation(10),  # 이미지를 -10도에서 10도 사이로 랜덤 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=normalized_channel_means, std=normalized_channel_stds),  # 채널별 평균과 표준편차로 정규화
])

# 배치 사이즈 설정
batch_size = 16  # 한 번에 처리할 데이터의 개수
'''Normalize  sms  mean으로 빼고  std 로 나누어 줍니다.'''

from cls import AugmentedDataset, CustomDataset
train_dataset = CustomDataset(train_df, transform=transform)
augmented_dataset = AugmentedDataset(train_df, transform=augment_transform)
combined_train_dataset = ConcatDataset([train_dataset, augmented_dataset])



print('train_dataset', len(train_dataset))
print('augmented_dataset', len(augmented_dataset))
print('combined_train_dataset', len(combined_train_dataset))

# import matplotlib.pyplot as plt  
# labels = ['train_dataset', 'valid_df', 'test_df' ]
# values_ori = [len(combined_train_dataset), len(valid_df), len(test_df) ]

# x = np.arange(len(labels))  # x 위치
# width = 0.35  # 막대의 너비
# # 비교 막대그래프 그리기
# fig, ax = plt.subplots()
# # 첫 번째 막대 (data1)
# rects1 = ax.bar(x - width/2, values_ori, width, label='train')
# # 두 번째 막대 (data2)
# # rects2 = ax.bar(x + width/2, values_test, width, label='test')
# # 그래프 레이블 설정
# ax.set_xlabel('Categories')
# ax.set_ylabel('Values')
# ax.set_title('Comparison of Two Data Sets')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# # 범례 추가
# ax.legend()
# plt.show()


# ddf = concat_dataset_to_dataframe(combined_train_dataset)
# show_bar_01(ddf,test_df)



if __name__ == "__main__":
    # CustomDataset 인스턴스 생성 (기본 학습 데이터셋)
   
    
    # DataLoader 생성 (학습, 검증, 테스트 데이터셋)
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 학습 데이터 로더 (데이터 섞기 활성화)
    valid_loader = DataLoader(CustomDataset(valid_df, transform=transform), batch_size=batch_size, shuffle=True, num_workers=4)  # 검증 데이터 로더
    test_loader = DataLoader(CustomDataset(test_df, transform=transform), batch_size=batch_size, shuffle=False, num_workers=4)  # 테스트 데이터 로더 (데이터 섞기 비활성화)

    # 사전 학습된 ResNet18 모델 불러오기
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

    # 모델의 마지막 완전 연결 계층(fc)을 사용자 정의 계층으로 대체
    # 이 계층은 in_features에서 10개의 출력으로 매핑합니다 (10개 클래스 분류를 위함)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 5),
    )

    class_counts = train_df['label'].value_counts().sort_index().values
    print(class_counts)
    
    augmented_labels = [1, 2, 3, 4]
    class_counts[augmented_labels] = class_counts[augmented_labels] *1
    
    
    weights = [1 / class_count for class_count in class_counts]
    class_weights = torch.FloatTensor(weights).cuda()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # 전체 훈련 횟수 설정
    num_epochs = 20
    # 최고 검증 정확도 초기화
    best_val_acc = 0.0
    # 얼리 스타핑을 위한 조건 설정 (성능 향상이 없을 때 몇 에포크까지 기다릴지)
    patience = 5
    # 연속적으로 성능 향상이 없는 에포크 수를 추적
    no_improve = 0

    # 훈련 및 검증 손실을 추적하기 위한 리스트
    train_losses = []
    valid_losses = []

    # 정해진 훈련 횟수만큼 반복
    for epoch in range(num_epochs):

        model.train() # 모델을 훈련 모드로 설정
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 훈련 데이터 로더를 통해 배치를 반복
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # 이전 반복에서 계산된 그래디언트를 초기화
            outputs = model(inputs) # 모델에 입력을 전달하여 출력을 계산
            loss = criterion(outputs, labels) # 손실 함수를 사용하여 손실 계산
            loss.backward() # 손실에 대한 그래디언트를 계산
            optimizer.step() # 옵티마이저를 사용하여 모델의 가중치를 업데이트

            running_loss += loss.item()  # 총 손실을 누적

            _, predicted = torch.max(outputs.data, 1) # 예측 결과 계산
            total_train += labels.size(0) # 전체 레이블 수 업데이트
            correct_train += (predicted == labels).sum().item() # 정확한 예측 수 업데이트

        # 에포크별 훈련 정확도 및 손실 계산
        train_acc = correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # Validate
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val
        val_loss = running_val_loss / len(valid_loader)

        # 손실 기록
        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # 최고 검증 정확도를 갱신하고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_resnet18.pth')
            no_improve = 0
        else:
            no_improve += 1 # 성능 향상이 없으면 no_improve 카운터 증가
            if no_improve >= patience:  # 설정한 얼리 스타핑 patience에 도달하면 학습을 중단합니다.
                print("Early stopping")
                break

# print('Finished Training')
# print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


