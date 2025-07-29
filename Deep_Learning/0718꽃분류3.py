# 250718 AM9시 
#케글꽃데이터셋 https://www.kaggle.com/datasets/alsaniipe/flowers-dataset
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data1\flowers2
# https://www.tensorflow.org/learn?hl=ko

from re import X
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import PIL.Image as pilimg
import os      
import imghdr
import numpy as np
import pandas as pd

# 데이터만들기 folder를 읽어서 데이터를 만들어보자 - train 폴더
base_path = "C:/Users/Admin/Documents/GitHub/python_workspace1N/Data_Analysis_2507/data1/flowers2"
def makeData(flower_name, label, isTrain=True):     # train/daisy 0   train/dandelion 1
    if isTrain:
        path = base_path + '/train/' + flower_name
    else:
        path = base_path + '/test/' + flower_name


    data = []
    labels = []
    # print(os.listdir(path))     # 해당 경로에 파일명을 모두 가져온다.
    # 파일 하나씩 읽어서 넘파이배열로 만들어서 data에 추가시키기
    i = 1
    for filename in os.listdir(path):
        try:
            if i % 100 == 0: #100개마다 출력
                print(f"{i}번째 file Processing: ", filename)
            i += 1
            # 파일 속성도 확인해보자
            kind = imghdr.what(path + "/" + filename)
            if kind in ['gif', 'png', 'jpeg', 'jpg']:   # 이미지일 때만
                img = pilimg.open(path + "/" + filename)    # 파일을 읽어서 numpy 배열로 바꾼다.
                resize_img = img.resize( (80, 80) )  # 사이즈는 특성이 너무 많으면 계산시간도 오래 걸리고
                                                    # 크기가 각각이면 학습 불가능. 그래서 적당한 크기를 맞춘다.
                pixel = np.array(resize_img)
                if pixel.shape == (80, 80, 3):
                    data.append(pixel)
                    labels.append(label)
        except:
            print(filename + " error")

    title = "train"
    if not isTrain:
        title = "test"
    savefileName = "imagedata{}.npz".format(str(label) + "_" + title)
    np.savez(savefileName, data=data, targets=labels)

def initData():
    flowers = ["daisy", "dandelion"]
    i = 0
    for f in flowers:
        makeData(f, i, True)
        makeData(f, i, False)
        i += 1

# npz 파일을 읽어서 넘파이 배열로 변환
def loadData():
    # train 데이터 합치기
    f1 = np.load("imagedata0_train.npz")
    f2 = np.load("imagedata1_train.npz")

    d1 = f1['data']  # data
    l1 = f1['targets']  # labels
    d2 = f2['data']  # data
    l2 = f2['targets']  # labels

    X_train = np.concatenate((d1, d2), axis=0)  # 두 개의 데이터 합치기
    y_train = np.concatenate((l1, l2), axis=0)  # 두 개의 레이블 합치기

    # test 데이터도 합치기
    f1 = np.load("imagedata0_test.npz")
    f2 = np.load("imagedata1_test.npz")
    d1 = f1['data']  # data
    l1 = f1['targets']  # labels
    d2 = f2['data']  # data
    l2 = f2['targets']  # labels
    X_test = np.concatenate((d1, d2), axis=0)  # 두 개의 데이터 합치기
    y_test = np.concatenate((l1, l2), axis=0)  # 두 개의 레이블 합치기
    return X_train, y_train, X_test, y_test

from tensorflow.keras import layers, models
def createModel():
    network = models.Sequential(
        [
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # 이진분류이므로 1개 클래스, sigmoid 활성화함수 사용
            # 마지막층은 이진분류이므로 2개 클래스, softmax 활성화함수 사용
            #암분류는 target이 악성0, 양성1(1일확률을 출력->예측확률)
            #target만 0,1로 바꾸고 2,softmax 활성화함수 사용
            #loss이진분류는 binary_crossentropy, 다중분류는 sparse_categorical_crossentropy
        ]
    )

    network.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",  # 이진분류이므로 binary_crossentropy 사용
        metrics=["accuracy"]
    )
    return network

from sklearn.preprocessing import StandardScaler
def preprocessing():
    X_train, y_train, X_test, y_test = loadData()
    # 데이터 전처리: 0-255 범위를 0-1로 정규화
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 데이터 차원 변경: (num_samples, height, width, channels) -> (num_samples, height * width * channels)
    X_train = X_train.reshape(X_train.shape[0], 80 * 80 * 3)  # 80x80 크기의 이미지
    X_test = X_test.reshape(X_test.shape[0], 80 * 80 * 3)

    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled

def main():
    # 데이터 불러오기
    X_train, y_train, X_test, y_test = loadData()
    print("Data loaded")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # 전처리: 0~255 -> 0~1 정규화
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 이미지 데이터 4차원 -> 2차원으로 변경 (Flatten)
    X_train = X_train.reshape(X_train.shape[0], -1)  # (num_samples, 80*80*3)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Preprocessing done")

    # 모델 생성
    network = createModel()
    print("Model created")

    # 학습 시작 직전
    print("Start training")
    history = network.fit(X_train_scaled, y_train, epochs=10, batch_size=100, verbose=1)

    # 학습 완료 후 평가
    print("Training completed. Evaluating model...")
    train_loss, train_acc = network.evaluate(X_train_scaled, y_train, verbose=1)
    print(f"훈련셋 손실값: {train_loss:.4f}, 정확도: {train_acc:.4f}") #소수점4자리까지만출력

    test_loss, test_acc = network.evaluate(X_test_scaled, y_test, verbose=1)
    print(f"테스트셋 손실값: {test_loss:.4f}, 정확도: {test_acc:.4f}")


if __name__ == "__main__":
    main()
    # initData() #데이터 초기화(한번만 실행하고 주석처리)
    X_train, y_train, X_test, y_test = loadData()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    # print("X_train:", X_train[:20])  # 첫 20개 이미지 출력
    # print("y_train:", y_train[:20])  # 첫 20개 레이블
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

""" 
# if __name__ == "__main__":
    # main() 결과 아래
Epoch 10/10
13/13 [==============================] - 0s 28ms/step - loss: 0.2843 - accuracy: 0.8824
Training completed. Evaluating model...
40/40 [==============================] - 1s 11ms/step - loss: 0.2307 - accuracy: 0.9129
훈련셋 손실값: 0.2307, 정확도: 0.9129
6/6 [==============================] - 0s 6ms/step - loss: 1.1612 - accuracy: 0.6319
테스트셋 손실값: 1.1612, 정확도: 0.6319
X_train shape: (1275, 80, 80, 3)
y_train shape: (1275,)
X_test shape: (182, 80, 80, 3)
y_test shape: (182,)
"""

"""
# if __name__ == "__main__":
#     initData() 결과 아래
# 100번째 file Processing: 
#
"""
