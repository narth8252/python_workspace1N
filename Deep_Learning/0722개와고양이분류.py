#250722 AM9시-CNN 쌤PPT415p (250717딥러닝종합_백현숙)
# 폴더에 train, test, validation 등으로 나눈다음에 각자에 폴더에 라벨을 만들고
#데이터 넣어놓고, ImageDataGenerator 나 DataSet 을 통해서 파일을 직접 읽어서
#학습한다. 데이터 증강 ImageDataGenerator(초창기부터)-폴더로부터 직접 이미지 파일
#을 읽어서 각종 처리를 해서 원하는 만큼 데이터를 늘려서 가져온다.
#좀더 정밀하게 비슷한 일을 한다. DataSet| - Tensor 2.xx이후 추가
#이미지 => numpy배열로 바꿔서 학습:데이터가 충분히 많으면

#개와 고양이 이미지 분류
#데이터셋 작을때 이미 학습된 CNN하고 작업시 사용예정
#  폴더생성해서 train, test, validation용 사진이동(노가다가 우리가 할일,중요함)
#1. cats_and_dogs_small > train, test, validation 생성
#   train>(/)cat 고양이사진1천장, /dog 개사진1천장 이동
#   test>(/) cat 고양이사진5백장, /dog 개사진5백장 이동
#   validation/cat 고양이사진5백장, /dog 개사진5백장 이동

import keras.utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras import models, layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import random 
import PIL.Image as pilimg 
import imghdr
import pandas as pd 
import pickle 
import keras 
import os
import shutil

#원본데이터셋이 있는 위치 경로
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\cats_and_dogs\train
# 현위치:\Data_Analysis_2507\DeepLearning
original_dataset_dir = "../data/cats_and_dogs/train" 

#이동위치 - 기본 폴더
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\cats_and_dogs_small
base_dir = "../data/cats_and_dogs_small"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test' )
validation_dir = os.path.join(base_dir, 'validation')

#ImageDataGenerator나 DataSet이나, 두 폴더보고 자동라벨링
train_cats_dir = os.path.join(train_dir, 'cats' )
train_dogs_dir = os.path.join(train_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats' )
test_dogs_dir = os.path.join(test_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# 케라스(keras) 모델 저장과 예측하기, 히스토리 저장법
#학습모델(네트워크)- 학습완료한 모델저장후 가져와서 예측
model_save_path_keras = 'cat_and_dogs_model.keras' 
#확장자가 .h5 → .keras로 변경, 케라스가 지원
history_filepath = 'cat_and_dogs_history.bin'
#학습시마다 정확도, 손실값있이 저장해서 줌
#이값자체는 저장미지원, 보통은 pickl(.pkl)사용저장
#.history 자체로 저장하면 에러, history.history(히스토리안 히스토리)로 저장하면 정상

def ImageCopy():
    #디렉토리內 파일개수 알아내기(현재 사용되지 않음, 정보성)
    totalCount = len(os.listdir(original_dataset_dir))
    print("전체개수", totalCount)

    #반복실행 위해 디렉토리 삭제(기존에 있다면)
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True, onerror=None)

    #디렉토리 생성(기본base 및 하위)
    os.makedirs(base_dir) #base부터 만들고 시작
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(validation_dir)

    os.makedirs(train_cats_dir)
    os.makedirs(train_dogs_dir)
    os.makedirs(test_cats_dir)
    os.makedirs(test_dogs_dir)
    os.makedirs(validation_cats_dir)
    os.makedirs(validation_dogs_dir)

    #파일이동 로직수정
    #고양이 사진 복사: 옮길 파일명이 cat0.jpg, cat1.jpg ,,, cat1000.jpg
    fnames = [ f'cat.{i}.jpg' for i in range(1000)] 
    for fname in fnames:   #train(0-999): 1000장
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst) #1개씩복사 x 1천번 반복
    
    fnames = [ f'cat.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames: #validation(1000-1499): 500장
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst) #1개씩 복사
    # ['cat.1000.jpg', 'cat.1001.jpg', ..., 'cat.1498.jpg', 'cat.1499.jpg']

    fnames = [ f'cat.{i}.jpg' for i in range(1500, 2000)]
    for fname in fnames:  #test(1500-1999): 500장
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst) #1개씩 복사

    #옮길 파일명이 dog0.jpg, dog1.jpg ,,, dog1000.jpg
    fnames = [ f'dog.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst) #1개씩복사 x 1천번 반복
    
    fnames = [ f'dog.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst) #1개씩 복사
    # ['dog.1000.jpg', 'dog.1001.jpg', ..., 'dog.1498.jpg', 'dog.1499.jpg']

    fnames = [ f'dog.{i}.jpg' for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst) #1개씩 복사
# ImageCopy() #복사후 주석처리
# 기존 디렉토리 정리: ../data/cats_and_dogs_small이 이미 존재하면, 새로 시작하기 위해 제거됩니다.
# 새 디렉토리 생성: base_dir과 모든 하위 디렉토리(train, test, validation 및 해당 cats, dogs 폴더)가 생성됩니다.
# 이미지 복사:
# data  폴더이동
# ├── cats_and_dogs
# │   └── train
# │       ├── cat.0.jpg
# │       ├── cat.1.jpg
# │       └── ...
# └── cats_and_dogs_small
#     ├── train
#     │   ├── cats  (고양이 이미지 1000장)
#     │   └── dogs  (개 이미지 1000장)
#     ├── test
#     │   ├── cats  (고양이 이미지 500장)
#     │   └── dogs  (개 이미지 500장)
#     └── validation
#         ├── cats  (고양이 이미지 500장)
#         └── dogs  (개 이미지 500장)

#DataSet 사용하기(옛날식코딩이긴함)
def deeplearning():
    #데이터증강 파라미터(데이터셋 작을때 과대적합 방지) → 에포크마다 데이터를 조금씩 변형해서 데이터셋크기를 인위적으로 늘림
    data_augmentation = keras.Sequential(
            [
		            layers.RandomFlip("horizontal", input_shape=(180, 180, 3)), #이미지를 수평으로 무작위로 뒤집기.
                    layers.RandomRotation(0.1),   #이미지를 최대 2pi 라디안의10(36도)까지 무작위로 회전
                    layers.RandomZoom(0.1)      #이미지를 최대 10%까지 무작위로 확대하거나 축소합니다.
                ]
        )
    
    #CNN 모델 아키텍처
    model = models.Sequential()
    #이미지 스케일링
    model.add(layers.Rescaling(1./255))  #픽셀값[0, 255]범위에서 [0, 1]로 정규화. 신경망에 대한 일반적인 전처리단계
    model.add(data_augmentation)      #정의된 데이터 증강 레이어를 적용
    model.add(layers.Conv2D(32, (3,3), activation='relu')) 
    #필터크기가 3times3인 32개필터를 가진 2D합성곱레이어. 활성화함수로는 relu(Rectified Linear Unit).
    model.add(layers.MaxPooling2D(2,2)) #풀사이즈가2times2(2x2)인 최대풀링레이어. 특징맵의 공간차원을 줄여 모델이 입력의 작은변화에 더 강건하도록 도움
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2)) #합성곱 및 풀링프로세스를 반복하여 네트워크가 더 복잡한 특징을 학습가능하게 함
    model.add(layers.Flatten())  #2D특징맵을 1D벡터로 평탄화해 완전연결Dense레이어를위한 데이터준비
    model.add(layers.Dropout(0.5)) #학습중 각업뎃에서 입력유닛의50%를 무작위로 0으로 설정, 뉴런간의 복잡한 상호적응을 줄여 과적합방지
    model.add(layers.Dense(512, activation='relu')) #512개유닛과 relu활성화를 가진 완전연결Dense레이어
    model.add(layers.Dense(1, activation='sigmoid')) # 출력레이어,이진분류이므로 sigmoid 사용

    #모델컴파일
    model.compile(optimizer='adam',  #adam옵티마이저는 학습중 모델의 가중치를 업데이트하는데 일반적이고 효과적인 선택
                  loss="binary_crossentropy", #이진분류니 이진교차엔트로피 손실함수, 예측된확률과 실제 이진레이블간의 차이 측정
                  metrics=['accuracy']) #모델성능=정확도 기준 평가
    
    #데이터셋 로드(학습) - train_dir폴더로부터 이미지파일 읽어오기
    train_ds = keras.utils.image_dataset_from_directory( 
        train_dir,             #학습이미지가 포함된 dir지정
        validation_split=0.2,   #train_dir의 훈련셋을 훈련셋20:검증셋80%로 나눠 검증
        seed=123,               # seed를 사용하여 2:8분할 일관성 유지
        subset="training",  #분할분 중 학습training 또는 검증validation부분을 로드여부 지정
        image_size=(180,180), #이미지픽셀크기 조정 180x180
        batch_size=16  #이미지16개씩 batch묶음으로 로드
    )
    #참고: 현재 코드는 train_dir에서 분할하여 train_ds와 val_ds를 모두 로드. 이는 ImageCopy()에 의해 생성된 validation 디렉토리가 학습 중 모델 검증에 사용되지 않는다는 것을 의미합니다. 미리 분리된 validation_dir을 검증에 사용하려면, validation_split 및 subset 인수 없이 validation_dir에서 val_ds를 직접 로드해야 합니다.

    #데이터셋 로드(검증)
    val_ds = keras.utils.image_dataset_from_directory( 
        train_dir,
        validation_split=0.2,
        seed=123,
        subset="validation",
        image_size=(180,180),
        batch_size=16
    )

    #모델학습 fit메소드
    history = model.fit(train_ds,  #학습 데이터셋
                        validation_data=val_ds, #학습중 모델의 성능을 모니터링하여 과적합을 감지하는 데 사용되는 검증데이터셋
                        epochs=30) 
    
    #모델 저장하기
    try:
        model.save(model_save_path_keras) #학습된 Keras 모델은 cat_and_dogs_model.keras에 저장
        print("모델 저장 완료")
    except Exception as e:
        print(f"모델 저장중 오류 발생 {e}")

    #히스토리 저장(필요시)
    try:
        with open(history_filepath, 'wb') as file: #cat_and_dogs_history.bin에 저장
            pickle.dump(history.history, file)
        print("히스토리 저장 완료")
    except Exception as e:
        print(f"히스토리 저장중 오류 발생 {e}")

    # model.save(model_save_path_keras)#모델 저장하기
    # model.save("catanddog.keras") #.keras확장자로 모델저장
    
# deeplearning()

# 학습 히스토리 시각화
def drawCart():
    print("--- 저장된 모듈 불러오기 ---")
    try:
        load_model_keras = keras.models.load_model(model_save_path_keras)
        print("모델 호출 성공")
    except Exception as e:
        print(f"모델 로딩중 실패: {e}")

    print("히스토리 불러오기")
    try:
        with open(history_filepath, 'rb') as file:
            history = pickle.load(file)
            print("히스토리 로딩 성공")
    except Exception as e:
        print(f"히스토리 로딩중 실패 : {e}")

    #히스토리의 키값들 가져오기 - 에포크횟수만큼 list로 가져온다
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    #X축(에포크) 좌표값
    X = range(len(acc))

    plt.plot(X, acc, 'ro', label="Training accuracy")
    plt.plot(X, val_acc, 'b', label='Validation accuracy')
    plt.title("Training and Validation accuracy")
    plt.legend() #범례표시(우측기본)

    plt.figure() #새창으로 차트띄움
    plt.plot(X, loss, 'ro', label='Training loss')
    plt.plot(X, val_loss, 'b', label='Validation loss')
    plt.title("Trainig and Validation loss")
    plt.legend() #범례표시(우측기본)
    plt.show() #모두출력

#모델 예측
def Predict():
    load_model_keras = None     #학습된 모델 불러오기
    try:
        load_model_keras = keras.models.load_model(model_save_path_keras)
        print("모델 호출 성공")
    except Exception as e:
        print(f"모델 로딩중 실패: {e}")
        return

    #예측 데이터셋 로드
    val_ds = keras.utils.image_dataset_from_directory( 
        train_dir,
        validation_split=0.2,
        seed=123,
        subset="validation",
        image_size=(180,180),
        batch_size=16
    )

    print("--- 자동 라벨링 확인하기 ---")
    class_names = val_ds.class_names
    print(class_names) #폴더명 기반으로 ['cats', 'dogs']

    total_match_count = 0 #전체일치한개수
    total_samples_process = 0 #전체 처리개수
    max_samples_to_process = 500 #데이터를 500개만 예측
    #예측 루프: `val_ds`의 이미지 배치들을 반복
    for input_batch, labels_batch in val_ds:
        #val_ds가 폴더로부터 이미지파일 읽어오는데 batch_size만큼씩 읽어온다
        total_samples_process += len(labels_batch)
    # print(total_samples_process)

        #예측하기
        predictions_batch = load_model_keras.predict(input_batch, verbose=2)
        print(predictions_batch)

        #확률을 클래스 레이블로 변환(시그모이드출력에 따라 고양이 0, 개 1)
        for i in predictions_batch:
            if i>=0.5:
                print("개")
            else:
                print("고양이")
        # print(predicted_classes)
        #선택: 4
        # 모델 호출 성공
        # Found 2000 files belonging to 2 classes.
        # Using 400 files for validation.
        # --- 자동 라벨링 확인하기 ---
        # ['cats', 'dogs']
        # 1/1 - 0s - 453ms/step
        # [[0.5129856 ]    # 개 (1에가까우면)
        #  [0.16446646]    # 고양이 (0에 가까우면)
        #  [0.99132216]    # 개
        #  [0.13068011]    # 고양이
        #  [0.9697823 ]    # 개
        #  [0.63931817]    # 개
        #  [0.93372333]    # 개
        #  [0.08725409]    # 고양이
        #  [0.88186204]    # 개
        #  [0.75304776]    # 개
        #  [0.862337  ]    # 개
        #  [0.8830812 ]    # 개
        #  [0.607399  ]    # 개
        #  [0.9224868 ]    # 개
        #  [0.5469079 ]    # 개
        #  [0.1390551 ]]    # 고양이        

        #예측결과와 실제 레이블 비교
        #이진분류라서 결과값이 1개만 온다. 꽃분류는 이렇게 해도 되지만 고양이는 안됨
        #이진분류시 라벨이1인요소의 확률 전달
        #다중분류시 [0.1, 0.1, 0.6, 0.1, 0.1]
        # pridicted_calsses = np.argmax(predictions_batch, axis=1)
        predicted_calsses = (predictions_batch>0.5).astype(int) #0.5보다크면T,작으면F
        print("예측: ", predicted_calsses.flatten()) #차원
        print("라벨: ", labels_batch.numpy())   #tensor → numpy
        # break #여기서 예측한것 확인후 주석처리

        match_count = np.sum(predicted_calsses.flatten() == labels_batch.numpy())
        total_match_count += match_count
     
    # print(total_samples_process, len(labels_batch)) # 16 16
    print("전체 데이터개수", total_samples_process) # 16
    print("정답 데이터개수", total_match_count) # 
    print("오답 데이터개수", total_samples_process-total_match_count) 

    # 선택: 4
    # 모델 호출 성공
    # Found 2000 files belonging to 2 classes.
    # Using 400 files for validation.
    # --- 자동 라벨링 확인하기 ---
    # ['cats', 'dogs']
    # 1/1 - 0s - 210ms/step
    # [[0.5129856 ]
    # ...
    # 예측:  [1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1]
    # 라벨:  [1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1]
    # 1/1 - 0s - 96ms/step
    # [[0.38705036]   # 고양이
    # [0.78980595]    # 개
    # [0.25163677]    # 고양이
    # [0.10577456]    # 고양이
    # [0.6678745 ]    # 개
    # [0.99406433]    # 개
    # [0.9999997 ]    # 개
    # [0.7170931 ]    # 개
    # [0.5183464 ]    # 개
    # [0.8482667 ]    # 개
    # [0.06098574]    # 고양이
    # [0.36380434]    # 고양이
    # [0.6999855 ]    # 개
    # [0.86362666]    # 개
    # [1.        ]    # 개
    # [0.9881058 ]]   # 개
    # 예측:  [0 1 0 0 1 1 1 1 1 1 0 0 1 1 1 1]
    # 라벨:  [0 1 0 1 1 1 1 1 0 1 0 0 1 1 1 1]
    # 2025-07-23 11:42:40.222624: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
    # 전체 데이터개수 400
    # 정답 데이터개수 282
    # 오답 데이터개수 118

def main():
    while True:
        print("1. 파일복사")
        print("2. 학습")
        print("3. 차트")
        print("4. 예측")
        sel = input("선택: ")
        if sel=="1":
            ImageCopy()
        elif sel=="2":
            deeplearning()
        elif sel=="3":
            drawCart()
        elif sel=="4":
            Predict()
        else:
            break

if __name__=="__main__":
    main()

# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning>python 0722개와고양이분류.py
# 2025-07-22 10:46:10.528593: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-22 10:46:13.227044: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning\0722개와고양이분류.py:26: DeprecationWarning: 'imghdr' is deprecated and slated for removal in Python 3.13
#   import imghdr
# 2025-07-22 10:46:21.108580: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 2000 files belonging to 2 classes.
# Using 1600 files for training.
# Found 2000 files belonging to 2 classes.
# Using 400 files for validation.
# Epoch 1/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 58s 527ms/step - accuracy: 0.9904 - loss: 0.0311 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 2/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 50s 503ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 3/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 53s 521ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 4/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 51s 506ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 5/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 50s 500ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 6/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 50s 497ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 7/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 50s 496ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 8/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 480ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 9/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 47s 473ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 10/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 481ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00


# 최종 250722
# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning>python 0722개와고양이분류.py
# 2025-07-22 13:15:23.972770: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-22 13:15:27.747528: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning\0722개와고양이분류.py:26: DeprecationWarning: 'imghdr' is deprecated and slated for removal in Python 3.13
#   import imghdr
# C:\ProgramData\anaconda3\envs\mytensorflow\Lib\site-packages\keras\src\layers\preprocessing\tf_data_layer.py:19: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(**kwargs)
# 2025-07-22 13:15:39.127680: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 2000 files belonging to 2 classes.
# Using 1600 files for training.
# Found 2000 files belonging to 2 classes.
# Using 400 files for validation.
# Epoch 1/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 55s 501ms/step - accuracy: 0.9485 - loss: 0.0411 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 2/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 49s 487ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 3/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 49s 487ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 4/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 484ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 5/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 482ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 6/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 49s 494ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 7/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 49s 488ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 8/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 49s 488ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 9/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 49s 486ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 10/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 50s 495ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00

# 250723 AM10시
# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning>python 0722개와고양이분류.py                              
# 2025-07-23 09:28:46.383750: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-23 09:28:50.194956: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning\0722개와고양이분류.py:26: DeprecationWarning: 'imghdr' is deprecated and slated for removal in Python 3.13
#   import imghdr
# C:\ProgramData\anaconda3\envs\mytensorflow\Lib\site-packages\keras\src\layers\preprocessing\tf_data_layer.py:19: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(**kwargs)
# 2025-07-23 09:29:01.356244: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 2000 files belonging to 2 classes.
# Using 1600 files for training.
# Found 2000 files belonging to 2 classes.
# Using 400 files for validation.
# Epoch 1/30
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 57s 507ms/step - accuracy: 1.0000 - loss: 0.0282 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 2/30
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 50s 504ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# ...
# # Epoch 27/30
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 481ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 28/30
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 483ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.100/100 11100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 483ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 29/30
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 481ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# Epoch 30/30
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 48s 479ms/step - accuracy: 1.0000 - loss: 0.0000e+00 - val_accuracy: 1.0000 - val_loss: 0.0000e+00
# 모델 저장 완료
# 히스토리 저장 완료

""" 
# # 53#학습모델(네트워크)- 학습완료한 모델저장후 가져와서 예측 -쉬운설명
# 1. 모델 저장과 불러오기
#  • 모델 저장

# 학습이 끝난 딥러닝(신경망) 모델은 파일로 저장 가능

# 최근에는 파일 확장자를 .keras를 사용

# # model_save_path_keras = 'cat_and_dogs_model.keras'
# # model.save(model_save_path_keras)
# 과거에는 .h5(HDF5 형식)가 주로 사용됐지만, 지금은 케라스에서 .keras를 권장

#  • 저장한 모델 불러와서 예측하기

# 저장된 모델을 다시 불러와서 바로 예측(추론) 가능
# # from keras.models import load_model
# # model = load_model(model_save_path_keras)
# # predictions = model.predict(new_data)

# 2. 학습 과정 기록(history) 저장하기
#  • history란?

# 모델을 학습할 때 model.fit()을 호출하면, 학습 과정의 정확도, 손실 값 등이 담긴 history 객체가 반환됨

#  • history 객체의 실제 데이터

# history에는 여러 정보가 들어 있지만, 실제 정확도/손실 값 데이터는
# history.history(딕셔너리 형태)에 들어 있음

#  • 바로 저장하면 오류 발생
# history 객체 자체를 그대로 저장하면 에러가 발생
# 반드시 history.history만 따로 뽑아서 저장해야 함
# 예시 코드: 히스토리 저장 & 불러오기 (pickle 사용)

# # import pickle

# # 저장
# # with open('cat_and_dogs_history.bin', 'wb') as f:
# #     pickle.dump(history.history, f)

# # 불러오기
# # with open('cat_and_dogs_history.bin', 'rb') as f:
# #     loaded_history = pickle.load(f)

# # print(loaded_history['accuracy'])  # 예: 학습 시 epoch별 정확도값
#  • pickle라이브러리를 사용하면 손쉽게 파이썬 데이터(딕셔너리 등)를 파일로 저장/불러오기 가능

# 3. 핵심 정리
#  • 모델 저장: model.save('파일명.keras')

#  • 모델 불러오기: load_model('파일명.keras') → 바로 예측 가능

#  • history 저장: 반드시 history.history만 추출해서 pickle 등으로 저장해야 함.
# (전체 history 저장은 지원되지 않고, 시도하면 에러)

#  • 확장자: 모델은 .keras(또는 예전방식 .h5), 히스토리는 .bin/.pkl 등 아무거나 가능

# 🔑 한 문장 요약
# 학습한 모델은 .keras로 저장 후 바로 불러와 예측할 수 있고,
# 학습 과정 기록(history)은 history.history를 따로 pickle로 저장해야 에러 없이 활용할 수 있다.
"""