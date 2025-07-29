# 250718 PM3시 쌤
# https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\garbage\Garbageclassification
# #폴더명에 공백없어야 데이터가져옴
from turtle import title
import tensorflow as tf
from tensorflow import keras
import PIL.Image as pilimg
import os      
import imghdr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# 데이터만들기 folder를 읽어서 데이터를 만들어보자 - train 폴더
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\garbage\Garbageclassification
base_path = "C:/Users/Admin/Documents/GitHub/python_workspace1N/Data_Analysis_2507/data/garbage/Garbageclassification"
# base_path = "../data/garbage/Garbageclassification"
title = "trash"
current_path = "./딥러닝/garbage/"
#⭐python_workspace1N\Data_Analysis_2507\딥러닝\garbage 이폴더 만들어서 npz파일저장함

def makeData(flower_name, label, isTrain=True):     #
    if isTrain:
        path = base_path + '/' + flower_name
    else:
        path = base_path + '/' + flower_name

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

    savefileName = current_path + "imagedata{}.npz".format(str(label) + "_" + title)
    np.savez(savefileName, data=data, targets=labels)

def initData():
    #print(os.listdir(base_path))  # base_path에 있는 폴더명 출력
    foldername = os.listdir(base_path)
    i = 0
    for f in foldername:
        makeData(f, i, True)
        i += 1
    print("Train data created-데이터저장 완료")

# npz 파일을 읽어서 넘파이 배열로 변환
def loadData():
    foldername = os.listdir(current_path)
    print("폴더명:", foldername)
    dataList = []
    targetList = []

    for i in range(0,len(foldername)):
        f1 = np.load(f"imagedata{i}_trash.npz")
        dataList.append(f1["data"])
        targetList.append(f1["targets"])

    X = np.concatenate(tuple(dataList), axis=0)  # 두 개의 데이터 합치기
    y = np.concatenate(tuple(targetList), axis=0) 

    #⭐데이터분할-4개 순서 그대로 써야 에러안남
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)  # 70% train, 30% test   
    print("Train data shape:", X_train.shape, y_train.shape)
    print("Test data shape:", X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

from tensorflow.keras import layers, models
def createModel():
    network = models.Sequential(
        [
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(6, activation="softmax") #⭐생성된npz파일개수가6개
        ]
    )

    network.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return network

from sklearn.preprocessing import StandardScaler
def preprocessing():
    X_train, y_train, X_test, y_test = loadData()
    # 데이터 전처리: 0-255 범위를 0-1로 정규화
    # X_train = X_train.astype("float32") / 255.0
    # X_test = X_test.astype("float32") / 255.0

    # 데이터 차원 변경: (num_samples, height, width, channels) -> (num_samples, height * width * channels)
    X_train = X_train.reshape(X_train.shape[0], 80 * 80 * 3)  # 80x80 크기의 이미지
    X_test = X_test.reshape(X_test.shape[0], 80 * 80 * 3)

    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test

def main():
    # 데이터 불러오기
    X_train, y_train, X_test, y_test = preprocessing()
    print("Data loaded")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # 모델 생성
    network = createModel()
    print("Model created")

    # 학습 시작 직전
    print("Start training")
    network.fit(X_train, y_train, epochs=10, batch_size=100)
# batch_size = 100  #한번에 모델에 넣어주는 이미지 개수, 한번에다넣으면 메모리터짐. 너무작으면 느려.
# epochs = 10  #모든데이터이미지/배치사이즈 = 에폭크만큼 반복해서 반복해서 똑똑하게만듬


    # 학습 완료 후 평가
    print("Training completed. Evaluating model...")
    train_loss, train_acc = network.evaluate(X_train, y_train, verbose=1)
    print(f"훈련셋 손실값: {train_loss:.4f}, 정확도: {train_acc:.4f}") #소수점4자리까지만출력

    test_loss, test_acc = network.evaluate(X_test, y_test, verbose=1)
    print(f"테스트셋 손실값: {test_loss:.4f}, 정확도: {test_acc:.4f}")


if __name__ == "__main__":
    # initData() #데이터 초기화(한번만 실행하고 주석처리)
    main()
    # X_train, y_train, X_test, y_test = loadData()
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # # print("X_train:", X_train[:20])  # 첫 20개 이미지 출력
    # # print("y_train:", y_train[:20])  # 첫 20개 레이블
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

""" 
# if __name__ == "__main__":
    # main() 결과 아래
"""

"""
# if __name__ == "__main__":
#     initData() 결과 아래
"""
