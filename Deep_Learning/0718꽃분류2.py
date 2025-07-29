# 250718 AM9시 
#케글꽃데이터셋 https://www.kaggle.com/datasets/alsaniipe/flowers-dataset
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data1\flowers2
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
    # print(path)

# base_path = "../data1/flowers2"  # ← 여기도 올바르게 고쳐줘야 함
# def makeData(flower_name, label, isTrain=True):
#     if isTrain:
#         path = os.path.join(base_path, 'train', flower_name)
#     else:
#         path = os.path.join(base_path, 'test', flower_name)

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
            layers.Dense(2, activation="softmax")
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
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 데이터 차원 변경: (num_samples, height, width, channels) -> (num_samples, height * width * channels)
    X_train = X_train.reshape(X_train.shape[0], 80 * 80 * 3)  # 80x80 크기의 RGB 이미지
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
# Model created
# Start training
# Epoch 1/10
# 13/13 [==============================] - 2s 39ms/step - loss: 3.3925 - accuracy: 0.5537
# Epoch 2/10
# 13/13 [==============================] - 0s 34ms/step - loss: 1.4019 - accuracy: 0.6353
# Epoch 3/10
# 13/13 [==============================] - 0s 33ms/step - loss: 1.0975 - accuracy: 0.6918
# Epoch 4/10
# 13/13 [==============================] - 1s 49ms/step - loss: 0.9391 - accuracy: 0.7153
# Epoch 5/10
# 13/13 [==============================] - 1s 52ms/step - loss: 0.8698 - accuracy: 0.7263
# Epoch 6/10
# 13/13 [==============================] - 1s 41ms/step - loss: 0.8571 - accuracy: 0.7553
# Epoch 7/10
# 13/13 [==============================] - 1s 47ms/step - loss: 0.7580 - accuracy: 0.7765
# Epoch 8/10
# 13/13 [==============================] - 0s 35ms/step - loss: 0.6128 - accuracy: 0.7937
# Epoch 9/10
# 13/13 [==============================] - 1s 40ms/step - loss: 0.7499 - accuracy: 0.8078
# Epoch 10/10
# 13/13 [==============================] - 0s 28ms/step - loss: 0.4786 - accuracy: 0.8549
# Training completed. Evaluating model...
# 40/40 [==============================] - 1s 12ms/step - loss: 0.3305 - accuracy: 0.8722
# 훈련셋 손실값: 0.3305, 정확도: 0.8722
# 6/6 [==============================] - 0s 21ms/step - loss: 1.0400 - accuracy: 0.5934
# 테스트셋 손실값: 1.0400, 정확도: 0.5934
# X_train shape: (1275, 80, 80, 3)
# y_train shape: (1275,)
# X_test shape: (182, 80, 80, 3)
# y_test shape: (182,)
"""

"""
# if __name__ == "__main__":
#     initData() 결과 아래
# 100번째 file Processing:  8797114213_103535743c_m_jpg.rf.132646214f7e9bcbdda402ab688ca868.jpg
# X_train shape: (1275, 80, 80, 3)
# X_train: [[[[139 140 135]
#    [147 147 146]
#    [149 149 150]
#    ...
#    [159 158 160]
#    [153 153 150]
#    [152 152 150]]

#   [[136 137 132]
#    [144 144 143]
#    [151 151 152]
#    ...
#    [157 156 158]
#    [154 154 152]
#    [149 149 147]]

#   [[132 130 127]
#    [139 139 136]
#    [152 152 151]
#    ...
#    [156 156 157]
#    [153 153 151]
#    [143 143 139]]

#   ...

#   [[ 40  44  21]
#    [ 42  45  22]
#    [ 45  48  27]
#    ...
#    [128 124 122]
#    [126 122 120]
#    [127 123 121]]

#   [[ 43  46  25]
#    [ 44  47  26]
#    [ 47  49  29]
#    ...
#    [133 129 128]
#    [129 125 124]
#    [129 125 124]]

#   [[ 44  47  26]
#    [ 44  47  26]
#    [ 48  50  28]
#    ...
#    [135 131 130]
#    [132 128 127]
#    [130 126 125]]]


#  [[[123 123 111]
#    [103  98  94]
#    [101  90  91]
#    ...
#    [135 150 126]
#    [154 161 141]
#    [144 155 130]]

#   [[104  99 101]
#    [ 77  67  78]
#    [ 75  60  75]
#    ...
#    [124 136 121]
#    [123 129 116]
#    [120 130 111]]

#   [[ 90  78  93]
#    [ 74  60  80]
#    [ 73  59  80]
#    ...
#    [107 116 104]
#    [113 122 105]
#    [108 120 102]]

#   ...

#   [[ 25   9  40]
#    [ 25   9  40]
#    [ 27  12  42]
#    ...
#    [ 42  18  77]
#    [ 45  20  80]
#    [ 46  21  81]]

#   [[ 28  12  41]
#    [ 25   9  38]
#    [ 27   7  38]
#    ...
#    [ 45  19  78]
#    [ 48  21  82]
#    [ 48  21  82]]

#   [[ 26  10  39]
#    [ 26  10  39]
#    [ 24   8  37]
#    ...
#    [ 49  20  80]
#    [ 52  22  84]
#    [ 53  23  85]]]


#  [[[ 26  16  80]
#    [ 25  15  79]
#    [ 25  15  78]
#    ...
#    [ 94  98 144]
#    [ 60  54 115]
#    [ 71  53 101]]

#   [[ 27  17  78]
#    [ 25  15  76]
#    [ 25  15  77]
#    ...
#    [ 61  59 113]
#    [ 52  41 102]
#    [ 65  48  96]]

#   [[ 27  17  76]
#    [ 26  16  76]
#    [ 26  16  78]
#    ...
#    [ 36  24  79]
#    [ 62  49 112]
#    [ 68  53  96]]

#   ...

#   [[  8   7  24]
#    [ 21  10  45]
#    [ 53  34  89]
#    ...
#    [ 22  15  51]
#    [ 22  15  50]
#    [ 21  12  48]]

#   [[ 24  13  49]
#    [ 50  31  85]
#    [ 75  54 110]
#    ...
#    [ 21  11  47]
#    [ 25  13  49]
#    [ 36  21  57]]

#   [[ 47  26  81]
#    [ 68  47 109]
#    [ 79  61 110]
#    ...
#    [ 21  13  48]
#    [ 21  12  47]
#    [ 26  14  50]]]


#  ...


#  [[[119  95  91]
#    [119  96  92]
#    [122  96  91]
#    ...
#    [115 115  40]
#    [119 120  46]
#    [114 107  52]]

#   [[116  92  88]
#    [120  96  92]
#    [115  90  86]
#    ...
#    [124 120  42]
#    [114 115  42]
#    [113  95  66]]

#   [[104  80  77]
#    [100  76  73]
#    [115  91  89]
#    ...
#    [120 121  40]
#    [105 102  47]
#    [118  90  63]]

#   ...

#   [[ 72  75  35]
#    [105 110  52]
#    [118 127  41]
#    ...
#    [ 25  33  11]
#    [ 27  37  14]
#    [ 36  55   9]]

#   [[ 78  70  59]
#    [ 59  58  37]
#    [ 68  78  28]
#    ...
#    [ 44  50  26]
#    [ 29  37  16]
#    [ 37  56   9]]

#   [[152 149  65]
#    [ 80  79  47]
#    [ 54  61  27]
#    ...
#    [ 49  57  22]
#    [ 33  41  17]
#    [ 52  70  29]]]


#  [[[ 97  68  75]
#    [120  90  82]
#    [127 110  86]
#    ...
#    [131  98 105]
#    [125  89  92]
#    [100  73  81]]

#   [[102  67  68]
#    [130  94  87]
#    [148 139 110]
#    ...
#    [116  88 100]
#    [125  89  97]
#    [113  82  90]]

#   [[104  70  76]
#    [120  80  83]
#    [123 112  90]
#    ...
#    [130 101 109]
#    [139 107 111]
#    [128 101  98]]

#   ...

#   [[ 98 124  98]
#    [ 84 116  88]
#    [ 93 123  83]
#    ...
#    [ 96 102 109]
#    [ 51  46  44]
#    [ 28  12  20]]

#   [[113 145 110]
#    [ 83 119  85]
#    [ 75 104  65]
#    ...
#    [ 98 106 106]
#    [ 71  71  68]
#    [ 49  39  39]]

#   [[105 139  96]
#    [ 67  96  63]
#    [ 37  55  44]
#    ...
#    [ 94 106  98]
#    [ 75  76  83]
#    [ 56  48  53]]]


#  [[[195 194 194]
#    [189 183 192]
#    [183 135  64]
#    ...
#    [181 178 172]
#    [186 182 180]
#    [183 180 174]]

#   [[198 194 199]
#    [184 189 191]
#    [175 152 117]
#    ...
#    [185 181 177]
#    [186 182 180]
#    [182 179 173]]

#   [[195 192 195]
#    [185 186 190]
#    [179 165 153]
#    ...
#    [186 183 182]
#    [184 180 178]
#    [184 181 174]]

#   ...

#   [[109 106 107]
#    [ 95  93  92]
#    [ 13  13  14]
#    ...
#    [181 129   3]
#    [182 121   6]
#    [182 119   5]]

#   [[102  97  99]
#    [ 33  31  32]
#    [  0   0   0]
#    ...
#    [179 143  59]
#    [166 115   3]
#    [180 121   3]]

#   [[ 54  49  50]
#    [  0   0   0]
#    [  0   0   0]
#    ...
#    [180 155 116]
#    [165 125  48]
#    [175 121   4]]]]
# y_train: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# y_train shape: (1275,)
# X_test shape: (182, 80, 80, 3)
# y_test shape: (182,)
"""
