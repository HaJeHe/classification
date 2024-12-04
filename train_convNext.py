from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
from cls import CustomDataset, AugmentedDataset  # 기존에 정의된 CustomDataset, AugmentedDataset
from tqdm import tqdm
from PreP import get_gpu_info, show_bar_01, compute_overall_mean_std
import random
import numpy as np
import pandas as pd


def seed_everything(seed):
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch cuda
    torch.backends.cudnn.deterministic = True  # CuDNN
    torch.backends.cudnn.benchmark = False  # CuDNN benchmark
seed_everything(seed=32)



image_path = "d:/python_code/resnet18/dataset/digit_data"
asset_path = "d:/python_code/resnet18/assets"


origin_train_df = pd.read_csv(f"{image_path}/train_data.txt", names=["path"])
origin_train_df["label"] = origin_train_df["path"].str[0].astype(int)
origin_train_df["path"] = image_path + "/" + origin_train_df["path"]


test_df = pd.read_csv(f"{image_path}/valid_data.txt", names=["path"])
test_df["label"] = test_df["path"].str[0].astype(int)
test_df["path"] = image_path + "/" + test_df["path"]

# show_bar_01(origin_train_df,test_df)


    
channel_means = [107.0054197492284, 101.17760866898148, 84.9624026234568]
channel_stds = [29.888622375757215, 25.865585869030646, 20.95805035580827]
normalized_channel_means = [x / 255 for x in channel_means]
normalized_channel_stds = [x / 255 for x in channel_stds]
  
    
# 데이터 전처리 및 증강 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalized_channel_means, std=normalized_channel_stds),
])

augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상
    transforms.ToTensor(),
    transforms.Normalize(mean=normalized_channel_means, std=normalized_channel_stds),
])

# 데이터셋 준비
train_df, valid_df = train_test_split(origin_train_df, test_size=0.2, random_state=0, shuffle=True, stratify=origin_train_df["label"])

from cls import AugmentedDataset, CustomDataset
train_dataset = CustomDataset(train_df, transform=transform)
augmented_dataset = AugmentedDataset(train_df, transform=augment_transform)
combined_train_dataset = ConcatDataset([train_dataset, augmented_dataset])

batch_size = 16


if __name__ == "__main__":
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(CustomDataset(valid_df, transform=transform), batch_size=batch_size, shuffle=True, num_workers=4)

    # ConvNeXt 모델 불러오기
    model = models.convnext_small(weights='DEFAULT')

    # 마지막 classifier 수정 (5개 클래스)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 5)

    # 손실 함수 및 옵티마이저 정의
    class_counts = train_df['label'].value_counts().sort_index().values
    weights = [1 / class_count for class_count in class_counts]
    class_weights = torch.FloatTensor(weights).cuda()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # 장치 설정 (GPU 사용 가능 여부에 따라)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 훈련 설정
    num_epochs = 20
    best_val_acc = 0.0
    patience = 5
    no_improve = 0

    train_losses = []
    valid_losses = []

    # 훈련 루프
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # 검증 루프
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

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # 검증 정확도가 개선되면 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_convNext.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break
