# 250818 AM9시  #PyTorch 사용해 아이리스(Iris) 데이터셋 분류하는 신경망모델을 구축하고 학습
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
iris = load_iris()
X = iris.data
y = iris.target

#2.표준화
scaler = StandardScaler()
X = scaler.fit_transform(X)

#3. 학습데이터와 테스트데이터 쪼개기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#numpy 배열 -> PyTorch 타입 배열(텐서)로 변환하자
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 4. 데이터셋과 데이터로더 만들기
#numpy배열 → PyTorch 타입 배열(텐서)로 변환
#모델한테 전달하기 위해서 데이터 셋을 만들고 DataLoader에게 전달해야 한다
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset (X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. 모델 정의 → 클래스로 만든다. nn.Module이라는 클래스를 반드시 상속받아야함
#부모클래스가 하는일이 많을때 이런식으로 설계
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__() #부모클래스 초기화, 부모생성자호출
        #super가 부모뜻함. 2개의 매개변수전달     #부모 생성자 호출코드는 메서드의 맨처음에 와야함. 
        #신경망의 층 정의, 입력은닉층(iris는 4개의 특성을 갖는다)
        self.fc1 = nn.Linear(4, 16) #fc1은 그냥 변수임. 모델저장해둔다. nn.Linear(입력개수, 출력개수)
        self.fc2 = nn.Linear(16, 16) #은닉층
        self.fc3 = nn.Linear(16, 3) #출력층(출력결과가3이어야한다). 파이토치는 softmax자동적용이라 안씀

        self.active = nn.ReLU() #활성화함수 정의, ReLU함수 사용

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.active(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.fc3(x)
        return x

# 6. 모델 인스턴스 생성, 모델만들고 손실함수,옵티마이저 정의
model = IrisClassifier()
criterion = nn.CrossEntropyLoss() #손실함수 - 다중분류, 파이토치는 softmax제공(자동적용)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. 모델 학습
# epochs = 100
def train_model(epochs):
    #학습 손실(loss)과 정확도를 매 epoch마다 기록 & 시각화 (matplotlib 그래프) 추가
    train_losses = [] 
    train_accuracies = []

    for epoch in range(epochs):
        model.train()  # 학습 모드
        running_loss = 0
        correct = 0
        total = 0

        #train_loader 배치사이즈만큼
        for inputs, labels in train_loader: #현재 배치사이즈는 16개씩 가져온다
            optimizer.zero_grad() #옵티마이저 초기화
            outputs = model(inputs) #순전파, 가중치 계산중
            loss = criterion(outputs, labels) #손실값을 계산한다
            loss.backward() #오차의 역전파
            optimizer.step()

            #추가
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        #추가
        epoch_loss = running_loss / total 
        epoch_acc = 100 * correct / total 
        train_losses.append(epoch_loss) 
        train_accuracies.append(epoch_acc) #정확도 기록

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    print("Training complete.")
    
    # 학습 결과 시각화 ,손실과 정확도 그래프
    plt.figure(figsize=(12,5))

    # 손실 그래프
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss over Epochs')
    plt.legend()

    # 정확도 그래프
    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy over Epochs')
    plt.legend()

    plt.show()


# 8. 훈련셋 정확도 Validation Accuracy 훈련셋 & 테스트셋 정확도 평가
def evaluate_model():
    model.eval()  #모델을 평가 모드로 변경

    with torch.no_grad(): #그라디언트 계산을 하지 않음
        correct_train = 0 #훈련셋이 예측잘맞는 경우 카운트하기위한 변수
        total_train = 0 #전체개수
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) #출력값이 확률, np.argmax쓰듯이 젤높은확률 착지
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item() #예측값과 실제값이 같은경우 카운트
        accuracy_train = 100*correct_train / total_train
        print(f"훈련셋 정확도: {accuracy_train:.2f}%")
        
        correct_test = 0
        total_test = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
        accuracy_test = 100 * correct_test / total_test
        print(f"테스트셋 정확도: {accuracy_test:.2f}%")

if __name__ == "__main__":
    train_model(epochs=100)
    evaluate_model() # 훈련셋 + 테스트셋 정확도 확인

# (mypytorch) C:\Users\Admin\Documents\GitHub\python_workspace1N\PyTorch>python 0818iris.py
# Epoch 1/100, Loss: 1.0503, Accuracy: 37.50%
# Epoch 2/100, Loss: 0.7936, Accuracy: 69.17%
# ...
# Epoch 99/100, Loss: 0.0244, Accuracy: 98.33%
# Epoch 100/100, Loss: 0.0243, Accuracy: 99.17%
# Training complete.
# 훈련셋 정확도: 98.33%
# 테스트셋 정확도: 96.67%