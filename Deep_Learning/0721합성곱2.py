#250721 AM11시
#컬러그림도 numpy배열로 잘정리해줌
# 컬러 이미지로 비교하는 일반 딥러닝 vs CNN
# 1. CIFAR-10 데이터와 컬러 이미지 구조
# CIFAR-10은 32×32 크기의 3채널(RGB) 컬러 이미지로 구성된 데이터셋입니다.
# 각 이미지는 numpy 배열로 shape이 (32, 32, 3)이며, 훈련 데이터는 5만 장, 테스트 데이터는 1만 장입니다.

# 데이터셋 위치
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\cifar-10-batches-py
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\cifar-10-batches-py-target_archive

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#5만개의 훈련셋, 1만개의 테스트셋

# print(X_train.shape)  # (50000, 32, 32, 3)
# print(y_train.shape)  # (50000, 1)
# print(np.unique(y_train))  # [0 1 2 3 4 5 6 7 8 9] #카테고리 개수 확인

#이미지 출력코드
#이미지 여러개 보기
def imageShow2(train_images, row, col):
    plt.figure(figsize=(10,5))
    for i in range(row*col):
        plt.subplot(row, col, i+1)
        image = train_images[i]
        plt.imshow(image, cmap = plt.cm.binary)
    plt.show()

imageShow2(X_train, 5, 5)

# 일반 Dense 신경망용 전처리
# Flatten : cnn 아닐때 차원변경
X_train_flat = X_train.reshape(50000, 32*32*3) / 255
X_test_flat = X_test.reshape(10000, 32*32*3) / 255
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
# 스케일링
# X_train = X_train/255
# X_test = X_test/255
# 라벨은 원핫인코딩
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#이번에는 CNN아닌걸로
def make_model1():
    network = models. Sequential([
    layers.Dense(128, activation='relu'), #세번째인자 input_shape 입력차원지정, 현재설치버전은 삭제
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

    network.compile(
        optimizer='sgd', 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    return network

# print("---- 학습 시작하기 ----")
# network.fit(X_train, y_train, epochs=10, batch_size=100)
# #머신러닝 score함수 대신 평가
# train_loss, train_acc = network.evaluate(X_train, y_train)
# print("훈련셋 손실 {}, 정확도 {}".format(train_loss, train_acc))

    
def make_model2():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(32,32,3)), #스케일링
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
# def make_model2() 함수 설명
# 입력: 32×32 크기의 RGB 컬러 이미지, 즉 (32, 32, 3) 형태.
# Layer 구성:
# Conv2D(합성곱층): 이미지 특징 추출, 필터 개수 및 커널 사이즈 설정.
# MaxPooling2D(풀링층): 특성 맵 크기 축소, 연산량 및 과적합 방지.
# Flatten: 2차원 이미지를 1차원으로 펼침.
# Dense(완전연결층): 분류기 역할, 마지막에 클래스 수만큼 출력.
# 컴파일: 최적화(adam), 손실(categorical_crossentropy), 정확도(accuracy) 지정.

#호출
# 이 구조는 CIFAR-10 및 비슷한 구조의 컬러 이미지 분류에 널리 사용됩니다.
# 데이터 전처리(정규화, 원-핫 인코딩 등)가 끝난 후 바로 사용할 수 있습니다.
if __name__ == "__main__":
    # 일반 신경망 학습 (Dense)
    model1 = make_model1()
    model1.fit(X_train_flat, y_train_cat, epochs=10, batch_size=100)
    print("Dense 모델 요약:")
    model1.summary()

    # CNN 학습
    model2 = make_model2()
    model2.fit(X_train, y_train_cat, epochs=10, batch_size=100)
    print("CNN 모델 요약:")
    model2.summary()

# (mytensorflow) PS C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning> python 0721합성곱2.py       
# TensorFlow 환경 경고 무시해도됨. 성능최적화에 대한 표준메세지로 결과값에 문제없음.
# 2025-07-21 13:31:24.325718: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-21 13:31:27.672116: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-21 13:31:50.451244: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

# 2. Dense(일반 신경망) 학습결과
# Epoch 1/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 6s 7ms/step - accuracy: 0.2409 - loss: 2.0938  
# Epoch 2/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.3505 - loss: 1.8334  
# ... 
# Epoch 9/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.4544 - loss: 1.5488  
# Epoch 10/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.4618 - loss: 1.5354  
# Dense 모델 요약:
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape 출력 형태       ┃         Param # ┃ 파라미터수
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (100, 128)                  │         393,344 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (100, 128)                  │          16,512 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (100, 10)                   │           1,290 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 411,148 (1.57 MB)  #파라미터 수
#  Trainable params: 411,146 (1.57 MB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 2 (12.00 B)
# C:\ProgramData\anaconda3\envs\mytensorflow\Lib\site-packages\keras\src\layers\preprocessing\tf_data_layer.py:19: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(**kwargs)
#▼2. Dense(일반신경망) 학습결과 설명
# • 최종 훈련셋 정확도: 약 46.2%
# • 신경망 구조
# 입력: 32×32×3(3072) 크기 이미지를 1차원 벡터로 평탄화 후 사용
# 은닉층(Dense) 2개(128 units)
# 출력(Dense) 1개(10 units, softmax)
# 파라미터 수: 411,148개

# • 레이어 	 출력 형태	  파라미터 수
#   Dense	(100, 128)	393,344
#   Dense	(100, 128)	16,512
#   Dense	(100, 10)	1,290
#   Total		411,148

# 3. CNN(합성곱 신경망) 학습 결과
# Epoch 1/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 39s 70ms/step - accuracy: 0.1983 - loss: 2.1245
# Epoch 2/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 36s 72ms/step - accuracy: 0.3736 - loss: 1.7024  
# Epoch 3/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 40s 80ms/step - accuracy: 0.4571 - loss: 1.4916  
# Epoch 4/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 42s 84ms/step - accuracy: 0.5162 - loss: 1.3553  
# Epoch 5/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 45s 89ms/step - accuracy: 0.5431 - loss: 1.2791  
# Epoch 6/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 87ms/step - accuracy: 0.5684 - loss: 1.2157  
# Epoch 7/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 42s 85ms/step - accuracy: 0.5881 - loss: 1.1592  
# Epoch 8/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 43s 87ms/step - accuracy: 0.6079 - loss: 1.1048  
# Epoch 9/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 41s 82ms/step - accuracy: 0.6232 - loss: 1.0649  
# Epoch 10/10
# 500/500 ━━━━━━━━━━━━━━━━━━━━ 42s 84ms/step - accuracy: 0.6418 - loss: 1.0264  
# CNN 모델 요약:
# Model: "sequential_1"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ rescaling (Rescaling)                │ (None, 32, 32, 3)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d (Conv2D)                      │ (None, 30, 30, 64)          │           1,792 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 15, 15, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 13, 13, 32)          │          18,464 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 11, 11, 32)          │           9,248 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 5, 32)            │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_3 (Conv2D)                    │ (None, 3, 3, 32)            │           9,248 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 1, 1, 32)            │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 32)                  │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 256)                 │           8,448 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_4 (Dense)                      │ (None, 128)                 │          32,896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_5 (Dense)                      │ (None, 64)                  │           8,256 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_6 (Dense)                      │ (None, 10)                  │             650 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 178,006 (695.34 KB)
#  Trainable params: 89,002 (347.66 KB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 89,004 (347.68 KB)

#▼3. CNN(합성곱 신경망) 학습 결과
# • 최종 훈련셋 정확도: 약 64.2%
# • 신경망 구조
# 입력: (32, 32, 3) 컬러 이미지 유지
# Conv2D, MaxPooling2D를 반복하며 이미지의 공간적 특성 추출
# 여러 합성곱층과 완전연결층(Dense)
# 마지막 출력은 10개의 클래스(softmax)

# • 파라미터 수: 178,006개 (실제 Trainable params: 89,002)
# Conv2D 계층의 파라미터 수가 Dense에 비해 적당하며, 효율적으로 학습 가능

# • 레이어	        출력 형태	파라미터 수
# Conv2D (64)	(30, 30, 64)	1,792
# MaxPooling2D	(15, 15, 64)	0
# Conv2D (32)	(13, 13, 32)	18,464
# Conv2D (32)	(11, 11, 32)	9,248
# MaxPooling2D	(5, 5, 32)	    0
# Conv2D (32)	(3, 3, 32)	    9,248
# MaxPooling2D	(1, 1, 32)	    0
# Flatten	    (32)	        0
# Dense (256)	(256)	        8,448
# Dense (128)	(128)	        32,896
# Dense (64)	(64)	        8,256
# Dense (10)	(10)	        650
# Total	----------------------- 178,006

# 4. 성능 및 구조 비교
# •항목	            Dense(일반 신경망)	             CNN(합성곱 신경망)
# 입력 데이터 형태	  1차원 벡터(3072), 공간정보 무시	3차원(32×32×3), 공간정보 유지
# 최종훈련셋 정확도   약 46%	                         약 64%
# 총 파라미터수	     411,148	                     178,006 (Trainable: 89,002)
# 특징	           학습초기속도빠름,공간특성활용불가    이미지특징자동추출,보통높은성능

# 5. 추가안내 및 주의사항
# • Dense 모델은 이미지 공간적(위치) 특성을 활용하지 못하므로 성능이 상대적으로 낮습니다.
# • CNN은 합성곱 연산 중심 구조로 이미지에서 패턴, 형태 등 중요한 정보를 자동 추출하여 성능이 훨씬 우수합니다.
# • 경고 메시지는 환경 특성 안내로 무시해도 됩니다.
# • 딥러닝 입문 실습 시 위 두 결과를 비교하면, CNN이 이미지 데이터에 왜 더 적합한지 확인할 수 있습니다.
# • 이 구조와 결과를 통해 이미지 분류 문제에서는 CNN이 일반적으로 훨씬 강력하고 효율적임을 알 수 있습니다.
