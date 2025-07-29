# 250725 AM11시 와인분류(CNN말고 딥러닝). 수현.py
# 와인 데이터셋처럼 수치형 데이터를 다룰 때는 CNN(합성곱 신경망)을 사용하지 않고 일반적인 딥러닝 모델, 즉 완전 연결(Dense) 층으로만 구성된 신경망으로도 충분합니다.
# 왜 와인 데이터셋에 CNN이 필요 없을까요?
# CNN은 주로 이미지나 영상, 음성과 같이 공간적 또는 시간적 패턴이 중요한 데이터에 특화된 딥러닝 모델입니다.
# 이미지 데이터: CNN은 필터(커널)를 사용하여 이미지의 픽셀 간의 **공간적 관계(예: 선, 모서리, 질감)**를 학습합니다. 얼굴의 윤곽선이나 고양이의 귀 모양 같은 특징은 픽셀들의 특정 배열에서 나타나죠.
# 와인 데이터셋: 와인 데이터셋은 13가지 화학 성분(예: 알코올 함량, 말산, 재)과 같은 수치형 특성들로 구성되어 있습니다. 이 특성들은 서로 특정한 공간적 또는 시간적 배열을 가지지 않습니다. 각 특성 값 자체가 독립적인 의미를 가지며, 이 값들의 조합으로 와인 종류가 결정됩니다.

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

# 왜 와인 데이터셋에 CNN이 필요 없을까요?
# CNN은 주로 이미지나 영상, 음성과 같이 공간적 또는 시간적 패턴이 중요한 데이터에 특화된 딥러닝 모델입니다.
# • 이미지 데이터: CNN은 필터(커널)를 사용하여 이미지의 픽셀 간의 **공간적 관계(예: 선, 모서리, 질감)**를 학습합니다. 얼굴의 윤곽선이나 고양이의 귀 모양 같은 특징은 픽셀들의 특정 배열에서 나타나죠.
# • 와인 데이터셋: 와인 데이터셋은 13가지 화학 성분(예: 알코올 함량, 말산, 재)과 같은 수치형 특성들로 구성되어 있습니다. 이 특성들은 서로 특정한 공간적 또는 시간적 배열을 가지지 않습니다. 각 특성 값 자체가 독립적인 의미를 가지며, 이 값들의 조합으로 와인 종류가 결정됩니다.
# 따라서 와인 데이터와 같이 테이블 형태의 수치형 데이터에서는 픽셀의 공간적 관계를 학습하는 CNN의 장점이 발휘되지 않습니다. 대신, 각 특성 값의 중요도와 특성들 간의 복합적인 관계를 학습하는 완전 연결(Dense) 층이 훨씬 효율적이고 적합합니다. 제가 이전 코드에서 Conv2D 층을 제거하고 Dense 층만 사용하도록 수정한 이유가 바로 이것입니다.
# 결론적으로, 와인 데이터셋에 대한 분류를 위해서는 CNN 없이 일반적인 딥러닝 모델만으로 충분

# sklearn의 와인 데이터셋을 사용하여 이전에 개/고양이 분류에 사용했던 딥러닝 코드 구조를 살짝 수정하여 와인 분류를 해보겠습니다.
# 개/고양이 데이터셋과 와인 데이터셋의 가장 큰 차이점은 다음과 같습니다:
# 1.입력 데이터 형태
# • 개/고양이: 이미지 데이터 (2D 픽셀 배열, RGB 채널) -> CNN에 적합
# • 와인: 수치형 데이터 (1D 특징 벡터, 13개 특성) -> 일반적인 Dense(완전 연결) 층에 적합

# 2.출력 (클래스) 개수:
# • 개/고양이: 2개 (이진 분류)
# • 와인: 3개 (다중 분류)
# 이 차이점을 반영하여 모델 구조를 변경해야 합니다.

# 3.수정할 주요 부분:
# • 데이터 로딩: ImageDataGenerator 또는 image_dataset_from_directory 대신 sklearn.datasets.load_wine()을 사용합니다.
# • 모델 입력 층: 이미지에 사용했던 Conv2D 층 대신 수치형 데이터를 처리할 수 있는 Dense 층으로 시작합니다. input_shape도 이미지 크기 대신 피처(특성) 개수로 변경합니다.
# • 출력 층: 3개의 클래스를 분류해야 하므로 Dense(1, activation='sigmoid') 대신 Dense(3, activation='softmax')를 사용합니다.
# • 손실 함수: 이진 분류에 사용했던 binary_crossentropy 대신 다중 분류에 적합한 sparse_categorical_crossentropy 또는 categorical_crossentropy를 사용합니다. (와인 데이터셋의 타겟 레이블이 0, 1, 2와 같은 정수이므로 sparse_categorical_crossentropy가 편리합니다.)
# • 데이터 증강: 수치형 데이터에는 이미지 증강이 필요 없으므로 해당 레이어를 제거합니다.
# • 히스토리/모델 저장 경로: 와인 분류를 위한 새로운 파일 이름을 사용합니다.

# 주요 수정 내용 설명:
# 1.sklearn.datasets.load_wine 사용: 데이터를 불러오는 부분이 이미지 파일 폴더에서 읽어오는 대신 load_wine() 함수를 직접 호출하도록 변경되었습니다.
# 2.train_test_split 사용: 수치형 데이터는 이미지처럼 폴더로 분리할 수 없으므로, sklearn.model_selection.train_test_split 함수를 사용하여 데이터를 학습, 검증, 테스트 세트로 분할합니다. stratify=y 옵션은 각 클래스의 비율을 유지하면서 분할하도록 도와줍니다.
# 3.모델 아키텍처 변경:
# • layers.Rescaling, data_augmentation, layers.Conv2D, layers.MaxPooling2D는 이미지 처리에 특화된 층이므로 모두 제거했습니다.
# • 대신 Dense 층을 여러 개 쌓아 올린 형태로 변경했습니다. 첫 번째 Dense 층의 input_shape는 와인 데이터의 피처(특성) 개수(X_train.shape[1], 즉 13)로 설정됩니다.
# • 출력 층은 와인 종류가 3가지이므로 layers.Dense(3, activation='softmax')로 변경했습니다. softmax는 다중 분류에서 각 클래스에 대한 확률을 출력합니다.

# 4.손실 함수 변경: binary_crossentropy 대신 sparse_categorical_crossentropy를 사용합니다. sparse_categorical_crossentropy는 타겟 레이블이 정수(0, 1, 2)일 때 편리하게 사용할 수 있습니다.
# 5.에포크 수 조정: 와인 데이터셋은 이미지 데이터셋보다 작고 단순하므로, 에포크 수를 30에서 50으로 늘려 더 안정적인 학습을 유도했습니다. (데이터에 따라 이 값은 달라질 수 있습니다.)

# 6.Predict_wine 함수:
# • 테스트 데이터셋을 다시 train_test_split으로 분할하여 X_test, y_test를 가져옵니다.
# • model.evaluate()를 사용하여 전체 테스트 셋에 대한 손실과 정확도를 한 번에 평가할 수 있도록 추가했습니다.
# • 개별 샘플 예측 시 predict 결과가 확률 분포(예: [0.05, 0.90, 0.05])로 나오므로, np.argmax()를 사용하여 가장 높은 확률을 가진 클래스의 인덱스를 선택하도록 했습니다.
# • 결과 출력 시 와인 종류의 이름(wine.target_names)도 함께 출력하여 가독성을 높였습니다.
# 이 코드를 실행하시면 와인 데이터셋에 대한 딥러닝 분류 모델을 학습하고 평가할 수 있습니다.







