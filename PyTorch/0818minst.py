# 250818 AM11시 #PyTorch 사용해 아이리스(Iris) 데이터셋 분류하는 신경망모델을 구축하고 학습
from itertools import batched
from pyexpat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scipy import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 하이퍼파라미터 설정
batch_size = 64 #메모리 적어서 못갖고오면 
learning_rate = 0.001
num_epochs = 100
#GPU있으면 GPU쓰고, 없으면 CPU사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터 준비: 이미지 전처리를 위한 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),   #PIL 이미지를 PyTorch 텐서로 변환(0~1사이값)
    transforms.Normalize((0.5,), (0.5,)) #텐서를 평균0.5, 표준편차0.5로 정규화
])

# 3. 데이터셋 다운로드 및 로드
# 학습용 데이터셋
train_dataset = datasets.MNIST(
    root='./data/',       # 데이터 저장 경로
    train=True,          # 학습용
    download=True,       # 없으면 다운로드
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 테스트용 데이터셋
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. 완전연결신경망 만들기
class ImageClassifier(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=500, num_classes=10):
        #그림크기= 28 by 28
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #784 → 500
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes) #500 → 10개로
        # self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        #이미지를 1차원벡터로 만들어서 전달
        # x = x.reshape(x.size(0), -1)  # Flatten the input
        x = x.reshape(-1, 28*28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)   #(self.fc1(x))
        x = self.fc2(x)
        return x
# 모델 만들기
model = ImageClassifier()

# 5. 손실함수 및 optimizer(최적화 기법) 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 6. 모델 학습
def train_model(epochs=100):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # images, labels = images.to(device), labels.to(device)

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels) #손실계산

            # 역전파 및 최적화
            optimizer.zero_grad() #가중치 초기화
            loss.backward() #역전파
            optimizer.step() #가중치 업데이트
            if (i+1)%100==0:
                print(f"Epochs [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# 7. 모델 평가
def evaluate_model():
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100*correct/total
    print(f"테스트셋 정확도 Accuracy: {accuracy:.2f}%")

if __name__=="__main__":
    train_model()
    evaluate_model()

# Epochs [1/100], Step [100/938], Loss: 0.2880
# Epochs [1/100], Step [200/938], Loss: 0.2177
# Epochs [1/100], Step [300/938], Loss: 0.1717
# Epochs [1/100], Step [400/938], Loss: 0.2458
# Epochs [1/100], Step [500/938], Loss: 0.2464
# Epochs [1/100], Step [600/938], Loss: 0.1719
# Epochs [1/100], Step [700/938], Loss: 0.2341
# Epochs [1/100], Step [800/938], Loss: 0.3427
# Epochs [1/100], Step [900/938], Loss: 0.1019
# ...
# Epochs [99/100], Step [900/938], Loss: 0.0085
# Epochs [100/100], Step [100/938], Loss: 0.0027
# Epochs [100/100], Step [200/938], Loss: 0.1018
# Epochs [100/100], Step [300/938], Loss: 0.0000
# Epochs [100/100], Step [400/938], Loss: 0.1091
# Epochs [100/100], Step [500/938], Loss: 0.0000
# Epochs [100/100], Step [600/938], Loss: 0.0002
# Epochs [100/100], Step [700/938], Loss: 0.0001
# Epochs [100/100], Step [800/938], Loss: 0.0000
# Epochs [100/100], Step [900/938], Loss: 0.0000
# 테스트셋 정확도 Accuracy: 97.94%