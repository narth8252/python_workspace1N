#250818 PM2시 #PyTorch 사용해 회귀분류로 캘리포니아 주택 가격 데이터셋 분류하는 신경망모델을 구축하고 학습
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from re import X

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

housing = fetch_california_housing()
X = housing.data
y = housing.target

#데이터 표준화
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#stratify=y는 y값의 비율을 유지하면서 쪼개기: 라벨이 불균형할때 그 비율에 맞춰 나눠라

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1) #2진분류시 shape변경
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1) 

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class HousingClassifier(nn.Module):
    def __init__(self):
        super(HousingClassifier, self).__init__()
        self.input = nn.Linear(8, 64) #입력층(8개의 특성)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 32)  #(은닉층, 출력층)
        self.fc2 = nn.Linear(32, 1)
        self.output = nn.Linear(32, 1)  #출력층(결과값1개)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = HousingClassifier()
criterion = nn.MSELoss() # 회귀 문제를 위한 손실 함수 Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=100):
    model.train()  # 학습 모드
    loss_history = []  # 에폭별 손실 기록

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")


    print("학습 완료")
    return loss_history

# 학습 과정의 Loss 변화를 시각화해서 에폭별로 확인
def plot_loss(loss_history):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

#회귀는 평가방식은 다름
def evaluate_model():
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 기울기 계산 비활성화
        total_loss = 0
        total_samples = 0

        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_samples += inputs.size(0)
            total_loss += loss.item()* inputs.size(0)  # 배치 크기를 곱하여 총 손실 계산

        #MSE와 RMSE 계산 
        avg_mse = total_loss / total_samples
        rmse = np.sqrt(avg_mse)
        print(f'테스트셋 Loss: {total_loss / total_samples:.4f}')
        print(f'테스트셋 평균 MSE: {avg_mse:.4f}') #소수점4개까지
        print(f'테스트셋 RMSE: {rmse:.4f}')

if __name__ == "__main__":
    loss_history = train_model(epochs=100)
    evaluate_model()
    plot_loss(loss_history)

# Epoch 1/100, Loss: 1.0147
# Epoch 2/100, Loss: 0.4322
# Epoch 3/100, Loss: 0.3971
# Epoch 4/100, Loss: 0.3783
# Epoch 5/100, Loss: 0.3632
# Epoch 6/100, Loss: 0.3589
# Epoch 7/100, Loss: 0.3443
# Epoch 8/100, Loss: 0.3432
# Epoch 9/100, Loss: 0.3269
# Epoch 10/100, Loss: 0.3248
# ...
# Epoch 90/100, Loss: 0.2398
# Epoch 91/100, Loss: 0.2380
# Epoch 92/100, Loss: 0.2377
# Epoch 93/100, Loss: 0.2377
# Epoch 94/100, Loss: 0.2369
# Epoch 95/100, Loss: 0.2379
# Epoch 96/100, Loss: 0.2353
# Epoch 97/100, Loss: 0.2333
# Epoch 98/100, Loss: 0.2344
# Epoch 99/100, Loss: 0.2386
# Epoch 100/100, Loss: 0.2339
# 학습 완료
# 테스트셋 Loss: 0.2691
# 테스트셋 평균 MSE: 0.2691
# 테스트셋 RMSE: 0.5187