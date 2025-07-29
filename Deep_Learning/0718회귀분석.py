#250718 AM11시 딥러닝
#회귀,이진분류,다중분류
# C:\Users\Admin\.keras\datasets  #.keras숨김폴더: 용량부족하면 지워도되고, 작업할때마다 새로 저장됨

import re
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

#boston_housing 데이터셋은 1970년대 보스턴 주택 가격에 대한 데이터셋
#load_data 함수가 4개로 나눠줌
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print("X_train[:5] shape:", X_train[:5].shape) #X_train[:5] shape: (5, 13)
print("y_train[:5] shape:", y_train[:5].shape) #y_train[:5] shape: (5,)

# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz
# 57026/57026 [==============================] - 0s 0us/step
# X_train shape: (404 프레임, 13개항목)
# y_train shape: (404,)
# X_test shape: (102, 13)
# y_test shape: (102,)

#스케일링
from sklearn.preprocessing import Normalizer
normal = Normalizer()
X_train_scaled = normal.fit_transform(X_train)
X_test_scaled = normal.transform(X_test)

from tensorflow.keras import models, layers
def makeModel():
    model = models.Sequential(
    [
        layers.Dense(512, activation='relu', input_shape=(13,)),  # input_shape생략가능
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        #마지막회귀의경우 확률가져오는게 아니라 연산결과1개만 가져온다.
        layers.Dense(1)  # 회귀는 활성화함수 없음
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


network = makeModel()
history = network.fit(X_train_scaled, y_train, epochs=10, batch_size=100)
train_loss, train_mae = network.evaluate(X_train_scaled, y_train)
print(f"훈련셋 손실값: {train_loss}, mae: {train_mae}")
test_loss, test_mae = network.evaluate(X_test_scaled, y_test)
print(f"테스트셋 손실값: {test_loss}, mae: {test_mae}")





