# 250725 AM11시 와인분류(CNN말고 딥러닝). 수현.py
# 와인 데이터셋처럼 수치형 데이터를 다룰 때는 CNN(합성곱 신경망)을 사용하지 않고 일반적인 딥러닝 모델, 즉 완전 연결(Dense) 층으로만 구성된 신경망으로도 충분합니다.
# 왜 와인 데이터셋에 CNN이 필요 없을까요?
# CNN은 주로 이미지나 영상, 음성과 같이 공간적 또는 시간적 패턴이 중요한 데이터에 특화된 딥러닝 모델입니다.
# 이미지 데이터: CNN은 필터(커널)를 사용하여 이미지의 픽셀 간의 **공간적 관계(예: 선, 모서리, 질감)**를 학습합니다. 얼굴의 윤곽선이나 고양이의 귀 모양 같은 특징은 픽셀들의 특정 배열에서 나타나죠.
# 와인 데이터셋: 와인 데이터셋은 13가지 화학 성분(예: 알코올 함량, 말산, 재)과 같은 수치형 특성들로 구성되어 있습니다. 이 특성들은 서로 특정한 공간적 또는 시간적 배열을 가지지 않습니다. 각 특성 값 자체가 독립적인 의미를 가지며, 이 값들의 조합으로 와인 종류가 결정됩니다.

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
import pickle

#CNN과 차이 수정할부분
# 데이터로딩: `ImageDataGenerator` 또는 `image_dataset_from_directory` 대신 `sklearn.datasets.load_wine()`
# 모델입력층: 이미지에 사용했던 Conv2D층 대신 수치형데이터처리하는 Dense층으로 시작, `input_shape`도 이미지크기대신 feature개수로 변경
# 출력층: 3개의 클래스를 분류해야하므로 Dense(1,actication='sigmoid')대신 Dense(3,activation='softmax')
# 손실함수: 이진분류용 `binary_crossentropy`대신 다중분류용 `sparse_categorical_crossentropy` 또는 `categorical_crossentropy`사용, 와인데이터셋의 타겟레이블이 0,1,2

import keras.utils
from tensorflow.keras.models import load_model
from keras import models, layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import pickle
import keras
import os
from sklearn.datasets import load_wine # 와인 데이터셋 불러오기
from sklearn.model_selection import train_test_split # 데이터 분할 도구

# 케라스(keras) 모델 저장과 예측하기, 히스토리 저장법
model_save_path_keras = 'wine_classification_model.keras'
history_filepath = 'wine_classification_history.bin'

def deeplearning_wine():
    # 1. 와인 데이터셋 불러오기
    wine = load_wine()
    X, y = wine.data, wine.target # 피처(X)와 타겟(y) 분리

    # 2. 데이터 분할 (학습, 검증, 테스트)
    # 개/고양이 코드에서는 폴더 구조로 분리했지만, 여기서는 데이터를 직접 분할합니다.
    # 학습: 훈련셋/검증셋 8:2 구조로 나눠 검증. 이전에 train_ds와 val_ds가 train_dir에서 분할된 방식과 유사하게,
    # 여기서는 X_train에서 X_val을 다시 분할하여 사용합니다.
    # 실제 테스트 데이터는 X_test를 따로 분리합니다.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y) # 테스트 20%
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full) # 훈련/검증 8:2

    print(f"전체 피처(X) 형태: {X.shape}")
    print(f"전체 타겟(y) 형태: {y.shape}")
    print(f"학습 데이터 형태: {X_train.shape}")
    print(f"검증 데이터 형태: {X_val.shape}")
    print(f"테스트 데이터 형태: {X_test.shape}")
    print(f"클래스 이름: {wine.target_names}")

    # 3. 모델 구축
    model = models.Sequential()
    # 입력 층: 와인 데이터는 13개의 수치형 특성을 가지므로 input_shape는 (13,)
    # 이미지 스케일링, 데이터 증강 (RandomFlip, RandomRotation, RandomZoom)은 수치형 데이터에 불필요하므로 제거
    model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],))) # X_train.shape[1]은 피처 개수 (13)
    model.add(layers.Dropout(0.3)) # 드롭아웃 비율 조정 (과적합 방지)
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    # 출력 층: 와인 종류는 3가지 (0, 1, 2)이므로 3개의 유닛과 'softmax' 활성화 사용
    # softmax는 다중 클래스 분류에서 각 클래스에 속할 확률을 출력합니다.
    model.add(layers.Dense(len(wine.target_names), activation='softmax'))

    # 4. 모델 컴파일
    model.compile(optimizer='adam',
                  # 타겟 레이블이 정수(0, 1, 2)이므로 sparse_categorical_crossentropy 사용
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    model.summary() # 모델 구조 확인

    # 5. 모델 학습
    history = model.fit(X_train, y_train,
                        epochs=50, # 에포크 수 조정
                        validation_data=(X_val, y_val),
                        verbose=2) # 학습 과정을 간결하게 출력

    # 6. 모델 저장
    try:
        model.save(model_save_path_keras)
        print("모델 저장 완료")
    except Exception as e:
        print(f"모델 저장중 오류 발생 {e}")

    # 7. 히스토리 저장
    try:
        with open(history_filepath, 'wb') as file:
            pickle.dump(history.history, file)
        print("히스토리 저장 완료")
    except Exception as e:
        print(f"히스토리 저장중 오류 발생 {e}")

def drawChart():
    print("--- 저장된 모듈 불러오기 ---")
    try:
        load_model_keras = keras.models.load_model(model_save_path_keras)
        print("모델 호출 성공")
    except Exception as e:
        print(f"모델 로딩중 실패: {e}")
        return

    print("히스토리 불러오기")
    try:
        with open(history_filepath, 'rb') as file:
            history = pickle.load(file)
            print("히스토리 로딩 성공")
    except Exception as e:
        print(f"히스토리 로딩중 실패 : {e}")
        return

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    X_epochs = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) # 1행 2열 중 첫 번째 플롯
    plt.plot(X_epochs, acc, 'bo', label="Training accuracy") # 'bo' for blue dots
    plt.plot(X_epochs, val_acc, 'b', label='Validation accuracy') # 'b' for solid blue line
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2) # 1행 2열 중 두 번째 플롯
    plt.plot(X_epochs, loss, 'ro', label='Training loss') # 'ro' for red dots
    plt.plot(X_epochs, val_loss, 'r', label='Validation loss') # 'r' for solid red line
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # 서브플롯 간 간격 자동 조절
    plt.show()

def Predict_wine():
    # 1. 학습된 모델 불러오기
    load_model_keras = None
    try:
        load_model_keras = keras.models.load_model(model_save_path_keras)
        print("모델 호출 성공")
    except Exception as e:
        print(f"모델 로딩중 실패: {e}")
        return

    # 2. 와인 데이터셋 불러와 테스트 데이터 준비
    wine = load_wine()
    X, y = wine.data, wine.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y) # 테스트 데이터만 다시 분할

    print(f"\n--- 테스트 데이터셋 형태: {X_test.shape} ---")
    print(f"클래스 이름: {wine.target_names}")

    # 3. 모델 평가 (선택 사항: 전체 테스트 셋에 대한 정확도 및 손실 확인)
    print("\n--- 모델 테스트셋 평가 ---")
    loss, accuracy = load_model_keras.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.4f}")

    # 4. 개별 샘플 예측
    print("\n--- 개별 샘플 예측 예시 (랜덤 5개) ---")
    random_indices = random.sample(range(len(X_test)), min(5, len(X_test)))

    total_match_count = 0
    
    for i in random_indices:
        sample_X = X_test[i:i+1] # 모델 입력 형태에 맞게 2D 배열로 만듭니다. (1, 피처개수)
        true_label = y_test[i]
        
        # 예측 수행
        predictions = load_model_keras.predict(sample_X, verbose=0) # verbose=0으로 출력 생략
        
        # 확률 출력 (softmax의 결과)
        predicted_probabilities = predictions[0]
        
        # 가장 높은 확률을 가진 클래스 선택
        predicted_class_index = np.argmax(predicted_probabilities)
        
        # 결과 출력
        print(f"\n샘플 {i}의 예측:")
        print(f"  예측 확률 (각 클래스): {predicted_probabilities}")
        print(f"  예측된 와인 종류 (인덱스): {predicted_class_index} -> {wine.target_names[predicted_class_index]}")
        print(f"  실제 와인 종류 (인덱스): {true_label} -> {wine.target_names[true_label]}")

        if predicted_class_index == true_label:
            print("  예측 일치! :)")
            total_match_count += 1
        else:
            print("  예측 불일치! :(")

    print(f"\n랜덤 {len(random_indices)}개 샘플 중 정답 개수: {total_match_count} / {len(random_indices)}")


def main():
    while True:
        print("\n--- 와인 분류 메뉴 ---")
        print("1. 학습")
        print("2. 차트")
        print("3. 예측")
        print("9. 종료")
        sel = input("선택: ")
        if sel=="1":
            deeplearning_wine()
        elif sel=="2":
            drawChart()
        elif sel=="3":
            Predict_wine()
        elif sel=="9":
            break
        else:
            print("잘못된 선택입니다. 다시 입력해주세요.")

if __name__=="__main__":
    main()

#     선택: 1 
# 전체 피처(X) 형태: (178, 13)
# 전체 타겟(y) 형태: (178,)
# 학습 데이터 형태: (113, 13)
# 검증 데이터 형태: (29, 13)
# 테스트 데이터 형태: (36, 13)
# 클래스 이름: ['class_0' 'class_1' 'class_2']
# C:\ProgramData\anaconda3\envs\mytensorflow\Lib\site-packages\keras\src\layers\core\dense.py:93: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.   
#   super().__init__(activity_regularizer=activity_regularizer, **kwargs)
# Model: "sequential_1"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense_4 (Dense)                      │ (None, 512)                 │           7,168 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout_2 (Dropout)                  │ (None, 512)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_5 (Dense)                      │ (None, 256)                 │         131,328 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout_3 (Dropout)                  │ (None, 256)                 │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_6 (Dense)                      │ (None, 128)                 │          32,896 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_7 (Dense)                      │ (None, 3)                   │             387 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 171,779 (671.01 KB)
#  Trainable params: 171,779 (671.01 KB)
#  Non-trainable params: 0 (0.00 B)
# Epoch 1/50
# 4/4 - 2s - 532ms/step - accuracy: 0.3274 - loss: 36.9537 - val_accuracy: 0.4138 - val_loss: 17.9919
# Epoch 2/50
# 4/4 - 0s - 32ms/step - accuracy: 0.3186 - loss: 24.3155 - val_accuracy: 0.2759 - val_loss: 10.8515
# Epoch 3/50
# 4/4 - 0s - 33ms/step - accuracy: 0.3274 - loss: 17.3919 - val_accuracy: 0.3103 - val_loss: 13.4638
# Epoch 4/50
# 4/4 - 0s - 37ms/step - accuracy: 0.4513 - loss: 11.8153 - val_accuracy: 0.4138 - val_loss: 6.7389
# Epoch 5/50
# 4/4 - 0s - 34ms/step - accuracy: 0.4159 - loss: 11.7432 - val_accuracy: 0.3103 - val_loss: 7.2271
# Epoch 6/50
# 4/4 - 0s - 35ms/step - accuracy: 0.3717 - loss: 8.6221 - val_accuracy: 0.3793 - val_loss: 5.3210
# Epoch 7/50
# 4/4 - 0s - 36ms/step - accuracy: 0.3363 - loss: 8.4550 - val_accuracy: 0.5862 - val_loss: 0.9688
# Epoch 8/50
# 4/4 - 0s - 36ms/step - accuracy: 0.3982 - loss: 8.1665 - val_accuracy: 0.3103 - val_loss: 4.1228
# Epoch 9/50
# 4/4 - 0s - 39ms/step - accuracy: 0.4602 - loss: 7.0685 - val_accuracy: 0.3793 - val_loss: 2.0985
# Epoch 10/50
# 4/4 - 0s - 35ms/step - accuracy: 0.3717 - loss: 6.6413 - val_accuracy: 0.5172 - val_loss: 1.4612
# Epoch 11/50
# 4/4 - 0s - 35ms/step - accuracy: 0.4779 - loss: 5.3187 - val_accuracy: 0.3793 - val_loss: 1.9372
# Epoch 12/50
# 4/4 - 0s - 36ms/step - accuracy: 0.5221 - loss: 3.8654 - val_accuracy: 0.4138 - val_loss: 1.1927
# Epoch 13/50
# 4/4 - 0s - 34ms/step - accuracy: 0.3894 - loss: 5.5603 - val_accuracy: 0.5172 - val_loss: 1.2804
# Epoch 14/50
# 4/4 - 0s - 32ms/step - accuracy: 0.5044 - loss: 3.9843 - val_accuracy: 0.4138 - val_loss: 1.6087
# Epoch 15/50
# 4/4 - 0s - 42ms/step - accuracy: 0.4071 - loss: 4.9128 - val_accuracy: 0.3448 - val_loss: 1.9252
# Epoch 16/50
# 4/4 - 0s - 53ms/step - accuracy: 0.4513 - loss: 4.2500 - val_accuracy: 0.5517 - val_loss: 1.0955
# Epoch 17/50
# 4/4 - 0s - 38ms/step - accuracy: 0.4690 - loss: 4.0672 - val_accuracy: 0.5517 - val_loss: 1.2331
# Epoch 18/50
# 4/4 - 0s - 40ms/step - accuracy: 0.4602 - loss: 3.2683 - val_accuracy: 0.5517 - val_loss: 0.9565
# Epoch 19/50
# 4/4 - 0s - 54ms/step - accuracy: 0.4779 - loss: 3.9399 - val_accuracy: 0.5517 - val_loss: 1.1267
# Epoch 20/50
# 4/4 - 0s - 34ms/step - accuracy: 0.5487 - loss: 2.6822 - val_accuracy: 0.5862 - val_loss: 1.1445
# Epoch 21/50
# 4/4 - 0s - 34ms/step - accuracy: 0.4956 - loss: 3.5123 - val_accuracy: 0.5862 - val_loss: 1.2673
# Epoch 22/50
# 4/4 - 0s - 35ms/step - accuracy: 0.4248 - loss: 3.3558 - val_accuracy: 0.5517 - val_loss: 2.1763
# Epoch 23/50
# 4/4 - 0s - 41ms/step - accuracy: 0.4513 - loss: 4.4065 - val_accuracy: 0.5862 - val_loss: 1.1960
# Epoch 24/50
# 4/4 - 0s - 35ms/step - accuracy: 0.4867 - loss: 3.4720 - val_accuracy: 0.6207 - val_loss: 1.1156
# Epoch 25/50
# 4/4 - 0s - 34ms/step - accuracy: 0.4425 - loss: 2.8401 - val_accuracy: 0.5862 - val_loss: 1.1105
# Epoch 26/50
# 4/4 - 0s - 37ms/step - accuracy: 0.5133 - loss: 2.9382 - val_accuracy: 0.6207 - val_loss: 1.2133
# Epoch 27/50
# 4/4 - 0s - 35ms/step - accuracy: 0.5133 - loss: 2.6334 - val_accuracy: 0.5862 - val_loss: 1.1147
# Epoch 28/50
# 4/4 - 0s - 35ms/step - accuracy: 0.5398 - loss: 2.1287 - val_accuracy: 0.5862 - val_loss: 0.9985
# Epoch 29/50
# 4/4 - 0s - 33ms/step - accuracy: 0.5044 - loss: 2.5391 - val_accuracy: 0.6207 - val_loss: 0.9507
# Epoch 30/50
# 4/4 - 0s - 34ms/step - accuracy: 0.4956 - loss: 2.1214 - val_accuracy: 0.5862 - val_loss: 1.0709
# Epoch 31/50
# 4/4 - 0s - 37ms/step - accuracy: 0.5664 - loss: 2.2534 - val_accuracy: 0.5862 - val_loss: 0.9582
# Epoch 32/50
# 4/4 - 0s - 35ms/step - accuracy: 0.5398 - loss: 2.1276 - val_accuracy: 0.5862 - val_loss: 0.9516
# Epoch 33/50
# 4/4 - 0s - 40ms/step - accuracy: 0.5487 - loss: 1.8420 - val_accuracy: 0.5862 - val_loss: 0.8772
# Epoch 34/50
# 4/4 - 0s - 35ms/step - accuracy: 0.5133 - loss: 1.9220 - val_accuracy: 0.4483 - val_loss: 1.0003
# Epoch 35/50
# 4/4 - 0s - 45ms/step - accuracy: 0.5841 - loss: 1.8505 - val_accuracy: 0.5172 - val_loss: 0.8861
# Epoch 36/50
# 4/4 - 0s - 32ms/step - accuracy: 0.4513 - loss: 1.7653 - val_accuracy: 0.5517 - val_loss: 0.9967
# Epoch 37/50
# 4/4 - 0s - 35ms/step - accuracy: 0.5221 - loss: 1.7148 - val_accuracy: 0.5172 - val_loss: 1.1542
# Epoch 38/50
# 4/4 - 0s - 36ms/step - accuracy: 0.5664 - loss: 1.8000 - val_accuracy: 0.6207 - val_loss: 0.8623
# Epoch 39/50
# 4/4 - 0s - 37ms/step - accuracy: 0.5575 - loss: 1.7008 - val_accuracy: 0.6207 - val_loss: 0.9401
# Epoch 40/50
# 4/4 - 0s - 34ms/step - accuracy: 0.5398 - loss: 1.8700 - val_accuracy: 0.5172 - val_loss: 0.8773
# Epoch 41/50
# 4/4 - 0s - 37ms/step - accuracy: 0.5133 - loss: 2.2410 - val_accuracy: 0.5862 - val_loss: 0.8476
# Epoch 42/50
# 4/4 - 0s - 48ms/step - accuracy: 0.4779 - loss: 2.1139 - val_accuracy: 0.5862 - val_loss: 0.8725
# Epoch 43/50
# 4/4 - 0s - 45ms/step - accuracy: 0.5398 - loss: 1.6253 - val_accuracy: 0.6552 - val_loss: 0.8915
# Epoch 44/50
# 4/4 - 0s - 33ms/step - accuracy: 0.4602 - loss: 1.8708 - val_accuracy: 0.5862 - val_loss: 0.9190
# Epoch 45/50
# 4/4 - 0s - 33ms/step - accuracy: 0.5575 - loss: 1.6026 - val_accuracy: 0.6207 - val_loss: 0.8317
# Epoch 46/50
# 4/4 - 0s - 32ms/step - accuracy: 0.5929 - loss: 1.5715 - val_accuracy: 0.5862 - val_loss: 0.9045
# Epoch 47/50
# 4/4 - 0s - 36ms/step - accuracy: 0.5310 - loss: 1.6000 - val_accuracy: 0.4828 - val_loss: 1.1927
# Epoch 48/50
# 4/4 - 0s - 35ms/step - accuracy: 0.5664 - loss: 1.4579 - val_accuracy: 0.5517 - val_loss: 0.9674
# Epoch 49/50
# 4/4 - 0s - 35ms/step - accuracy: 0.5221 - loss: 1.4967 - val_accuracy: 0.5862 - val_loss: 0.9544
# Epoch 50/50
# 4/4 - 0s - 32ms/step - accuracy: 0.5664 - loss: 1.2083 - val_accuracy: 0.5172 - val_loss: 1.0368
# 모델 저장 완료
# 히스토리 저장 완료

# 선택: 3
# 모델 호출 성공

# --- 테스트 데이터셋 형태: (36, 13) ---
# 클래스 이름: ['class_0' 'class_1' 'class_2']

# --- 모델 테스트셋 평가 ---
# 테스트 손실: 0.7413
# 테스트 정확도: 0.6667

# --- 개별 샘플 예측 예시 (랜덤 5개) ---

# 샘플 18의 예측:
#   예측 확률 (각 클래스): [0.10995506 0.6014086  0.28863633]
#   예측된 와인 종류 (인덱스): 1 -> class_1
#   실제 와인 종류 (인덱스): 1 -> class_1
#   예측 일치! :)

# 샘플 1의 예측:
#   예측 확률 (각 클래스): [0.12976906 0.57552326 0.29470766]
#   예측된 와인 종류 (인덱스): 1 -> class_1
#   실제 와인 종류 (인덱스): 2 -> class_2
#   예측 불일치! :(

# 샘플 16의 예측:
#   예측 확률 (각 클래스): [0.0029842  0.6476777  0.34933808]
#   예측된 와인 종류 (인덱스): 1 -> class_1
#   실제 와인 종류 (인덱스): 1 -> class_1
#   예측 일치! :)

# 샘플 26의 예측:
#   예측 확률 (각 클래스): [0.89006746 0.09145826 0.01847424]
#   예측된 와인 종류 (인덱스): 0 -> class_0
#   실제 와인 종류 (인덱스): 0 -> class_0
#   예측 일치! :)

# 샘플 28의 예측:
#   예측 확률 (각 클래스): [0.02367662 0.71535903 0.2609644 ]
#   예측된 와인 종류 (인덱스): 1 -> class_1
#   실제 와인 종류 (인덱스): 1 -> class_1
#   예측 일치! :)

# 랜덤 5개 샘플 중 정답 개수: 4 / 5
