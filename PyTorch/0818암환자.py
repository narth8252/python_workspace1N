#250818 PM1시 #PyTorch 사용해 아이리스(Iris) 데이터셋 분류하는 신경망모델을 구축하고 학습
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from re import X
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

#데이터 표준화
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#stratify=y는 y값의 비율을 유지하면서 쪼개기: 라벨이 불균형할때 그 비율에 맞춰 나눠라

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1) #2진분류시 shape변경
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1) 
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CancerClassifier(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_classes=1):
        super(CancerClassifier, self).__init__()
        self.fc1 = nn.Linear(30,64)  #특성30 (input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,32)  #(은닉층, 출력층)
        self.fc3 = nn.Linear(32,1)  #(은닉층, 출력층) 결과값1개
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
        return x
    
model = CancerClassifier()
criterion = nn.BCEWithLogitsLoss() # 이진 분류를 위한 손실 함수: 시그모이드+Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("학습완료")

def evaluate_model():
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 기울기 계산 비활성화
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs)) #반올림이라 0또는1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'테스트셋 Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    train_model(100)
    evaluate_model()

# Epoch 1/100, Loss: 0.6096
# Epoch 2/100, Loss: 0.4361
# Epoch 3/100, Loss: 0.2230
# Epoch 4/100, Loss: 0.1450
# Epoch 5/100, Loss: 0.0370
# Epoch 6/100, Loss: 0.0858
# Epoch 7/100, Loss: 0.0278
# Epoch 8/100, Loss: 0.0272
# Epoch 9/100, Loss: 0.2107
# Epoch 10/100, Loss: 0.0219
# ...
# Epoch 90/100, Loss: 0.0000
# Epoch 91/100, Loss: 0.0044
# Epoch 92/100, Loss: 0.0001
# Epoch 93/100, Loss: 0.0008
# Epoch 94/100, Loss: 0.0001
# Epoch 95/100, Loss: 0.0000
# Epoch 96/100, Loss: 0.0018
# Epoch 97/100, Loss: 0.0000
# Epoch 98/100, Loss: 0.0006
# Epoch 99/100, Loss: 0.0003
# Epoch 100/100, Loss: 0.0000
# 학습완료
# 테스트셋 Accuracy: 95.61%