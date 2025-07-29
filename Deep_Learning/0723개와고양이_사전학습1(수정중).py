#250723 PM1시-CNN 쌤PPT415p (250717딥러닝종합_백현숙)
#0722개와고양이분류.py코드보다 간결.
# 투스테이지
#사전학습된 모델을 가져다 사용하기 (특성추출, 미세조정)
#1. 특성추출하기 - CNN하고 완전피드포워딩
#2. 이미 학습된 모델을 불러와서 CNN파트랑 완전연결망을 쪼개서
# CNN으로 부터 특성을 추출한 다음에 완전연결망한테 보내서 다시 학습을 한다
# CNN이 시간이 많이 걸린다. => CNN재활용을 하면 학습시간도 적게 걸리고, 예측률도 더 높아진다.
# 이미 수십만장의 사진을 가지고 학습한 모델을 갖다 쓴다.
#   장점) 데이터셋이 적을경우(1000장)
#        이미 학습된 모델을 사용함으로써 학습시간을 줄여준다
#        컴퓨터 자원이 작아도 학습가능(학습된모델이 용량이 더 큰경우도有)
# VGG-19 Tensorflow구현 참고 https://hwanny-yy.tistory.com/11 

# VGG19, ResNet, MobileLet 등 이미지셋(학습된) 모델들이 있다.
# (base) C:\Windows\System32>conda activate mytensorflow
# (mytensorflow) C:\Windows\System32>conda install gdown
import gdown
# gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output='dogs-vs-cats.zip')
#실행시키면 실행한 폴더에 다운로드됨 dogs-vs-cats.zip 이렇게 다운로드하는 방식도 있음.
#받고나서 주석처리.안그러면 재다운로드
# \Data_Analysis_2507\data\dogs-vs-cats

import os, shutil, pathlib
import keras.applications
import keras.applications.vgg19
import tensorflow as tf
import numpy as np
import keras
import pickle
import matplotlib.pyplot as plt # 이미지 시각화를 위한 라이브러리
from keras import models, layers 
from keras.utils import image_dataset_from_directory
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

original_dir = pathlib.Path("../data/dogs-vs-cats/train")
new_base_dir = pathlib.Path("../data/dogs-vs-cats/dogs-vs-cats_small")

#make_subset 함수: 데이터를 서브셋(train, validation, test)으로 분할하여 복사
def make_subset(subset_name, start_index, end_index): #make_subset("train", 0, 1000)
    print(f"\n--- {subset_name} 서브셋 생성 중 ({start_index}에서 {end_index-1}까지")
    for category in ("cat", "dog"):
        dir = new_base_dir/subset_name/category
        os.makedirs(dir, exist_ok=True) #디렉토리 없을경우 새디렉토리 생성
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir/fname, dst=dir/fname)
    print(f"--- {subset_name} 서브셋 생성 완료 ---")
# -- test 서브셋 생성 완료(한번만 실행) --
# make_subset("train", 0, 1000)
# make_subset("validation", 1000, 1500)
# make_subset("test", 1500, 2000)

# --- 데이터셋 로드 ---
# from keras. utils import image_dataset_from_directory
#batch_size에 지정된 만큼 폴더로부터 이미지를 읽어온다. 크기는 image_size에 지정한 값으로 가져온다
def load_image_datasets():
    train_ds = image_dataset_from_directory(
        new_base_dir/"train",
        image_size=(180,180),
        batch_size=16
    )
    validation_ds = image_dataset_from_directory(
        new_base_dir/"validation",
        image_size=(180,180),
        batch_size=16
    )
    test_ds = image_dataset_from_directory(
        new_base_dir/"test",
        image_size=(180,180),
        batch_size=16,
        shuffle=False # 테스트셋은 섞지 않아야 라벨과 순서 일치
    )
    print("--- 이미지 데이터셋 로드 완료 ---")
    return train_ds, validation_ds, test_ds

#VGG19 이미지모델 가져오기
# VGG19 컨볼루션 베이스 모델 로드 (전역 변수로 선언)
# 한 번만 로드하고 재사용하는 것이 효율적입니다.
from keras.applications.vgg19 import VGG19
conv_base = keras.applications.vgg19.VGG19( 
    weights = "imagenet",     # ImageNet 데이터셋으로 사전 학습된 가중치 사용
    include_top=False,        # 모델의 분류 헤드(top)를 제외하고 컨볼루션 베이스만 가져옴
    input_shape=(180, 180, 3) #⭐젤중요:입력할데이터크기(test_ds의 image_size일치)
)
conv_base.summary() #CNN요약확인
#block5_pool (MaxPooling2D)  (None, 5, 5, 512)

#데이터셋을 주로 CNN으로부터 특성(features)과 라벨(labels)을 추출 전달하는 함수
def get_features_and_labels(dataset):
    all_features=[]
    all_labels=[]
    print(f"\n--- {dataset.file_paths[0].split(os.sep)[-2]} 데이터셋에서 특성 추출 중 ---")
    # 'image_dataset_from_directory'가 자동으로 'labels'를 정수형으로 반환합니다.
    # VGG19 전처리 함수는 이미지를 0~255 범위에서 VGG19에 맞는 형태로 변환합니다.
    for images, labels in dataset:
        # VGG19가 학습될 때 사용된 것과 동일하게 이미지 전처리
        preprocessed_images = keras.applications.vgg19.preprocess_input(images)
        print(images.shape, preprocessed_images.shape)
        # --- ▼ 아래3줄 확인후 주척처리 ▼ ---
        # plt.imshow(preprocessed_images[0]) 
        # plt.show() 
        # break
        # --- ▼ 아래줄에서 특성 추출됨 ▼ ---
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels.numpy()) # 텐서를 NumPy 배열로 변환
    print(f"--- 특성 추출 완료 ---")
    return np.concatenate(all_features), np.concatenate(all_labels)

# 추출된 특성과 라벨을 파일로 저장하는 함수
def save_features():
    # 이미지 데이터셋 로드
    train_ds, validation_ds, test_ds = load_image_datasets()
    train_features, train_labels = get_features_and_labels(train_ds)
    validation_features, validation_labels = get_features_and_labels(validation_ds)
    test_features, test_labels = get_features_and_labels(test_ds)

    data = [train_features, train_labels, validation_features, validation_labels,
            test_features, test_labels]
    with open('개고양이특성.bin', 'wb') as file:
        pickle.dump(data, file)

def load_features():
    with open('개고양이특성.bin', 'rb') as file:
        data = pickle.load(file)

    return data[0], data[1], data[2], data[3], data[4], data[5] #튜플

def deeplearning():
    print("특성 추출 중 (더미 데이터 사용)...")
    # train_features, train_labels = get_features_and_labels(train_ds)
    # validation_features, validation_labels = get_features_and_labels(validation_ds)
    # test_features, test_labels = get_features_and_labels(test_ds)
    train_features, train_labels, validation_features, validation_labels , \
    test_features, test_labels= load_features()

    #특성추출, 불러오기, 예측
    # 모델(분류기) 정의
    data_augmentation = keras.Sequential(
        [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1)
            ]
    )

    #맨마지막 block
    inputs = keras.Input(shape=(5,5,512)) # VGG16마지막 Conv블록 출력shape에 맞춤
    x = data_augmentation(inputs) # Feature map에 데이터 증강적용 (선택사항, 주의필요)
                                  # 일반적으로는 이미지에 직접 증강적용
                                  # 여기서는 특징맵에 적용하도록 되어있어 그대로 둠.
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x) # 활성화 함수 추가
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    #과소/과대 무관하게 학습완료해야 저장가능
    # → 콜백함수(위 방지법): 과대적합되는 시점에 저장 → 콜백함수에 저장할 파일명 전달하면 자동호출
    #list형태로 받아가서 여러개 줄줄d이 쓸수있다는 말임
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath = '특성추출.keras',
            save_best_only=True, #최적시 저장
            monitor = "val_loss" #검증손실(val_loss)이 최소일 때 저장
        )
    ]
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary() 
    
    print("\n--- 분류 모델 훈련 시작 ---")
    history = model.fit(train_features, train_labels,
                        epochs=5, 
                        validation_data=(validation_features, validation_labels),
                        callbacks = callbacks)
    
#예측(test_features, test_labels 이걸로 해야 빠름)
def predict():
    model = keras.models.load_model("특성추출.keras")
    train_features, train_labels, validation_features, validation_labels , \
    test_features, test_labels= load_features()

    test_pred = model.predict(test_features)
    test_pred = (test_pred>0.5).astype("int").flatten()
    print(test_pred[ :20])
    print(test_labels[:20])
    match_count = np.sum(test_pred == test_labels)
    print("맞춘개수 : ", match_count)
    print("틀린개수 : ", len(test_labels)-match_count )

    # for i in range(0, 10):
    # print(test_labels[i], test_pred[i], test_labels[i] == test_pred[i])

def main():
    while True:
        print("1. 특성추출")
        print("2. 학습")
        print("3. 예측")
        sel = input("선택: ")
        if sel=="1":
            ImageCopy()
        elif sel=="2":
            deeplearning()
        elif sel=="3":
            Predict()
        else:
            break

if __name__=="__main__":
    main()

# deeplearning()

# 왜 특성 추출이라고 부를까요?
# 원래 이미지를 픽셀 단위로 입력하는 대신, 이미지를 통해 얻은 **더 추상적이고 의미 있는 정보(특징)**를 추출하여 다음 단계(완전 연결 분류기)의 입력으로 사용하기 때문입니다. 마치 사람이 사진을 보고 "이건 개 코 모양이야", "이건 고양이 귀 모양이야"와 같은 특징들을 인식하는 것과 비슷하다고 볼 수 있습니다. 🐶🐱
# 이렇게 추출된 특징은 원본 이미지보다 훨씬 더 정보 밀도가 높고 노이즈에 강하며, 다음 단계의 간단한 분류기가 이 특징을 바탕으로 '개'인지 '고양이'인지 쉽게 학습할 수 있게 됩니다.
# VGG19 모델을 사용하여 이미지를 "특성 추출(Feature Extraction)" 했습니다.

# 특성 추출이 일어난 곳 :핵심은 conv_base 객체
# 특성 추출은 get_features_and_labels 함수 내부에서 conv_base.predict(preprocessed_images) 라인을 통해 이루어집니다.
# 1. conv_base 정의: conv_base는 ImageNet이라는 방대한 데이터셋으로 이미 학습된 VGG19 모델의 컨볼루션 베이스를 불러온 것입니다. 여기서 include_top=False가 매우 중요한데, 이는 VGG19의 원본 분류기(ImageNet 1000개 클래스 분류용)를 제외하고, **이미지의 시각적 특징을 학습하는 부분(컨볼루션 레이어들)**만 가져오겠다는 의미입니다.
# 2-1. preprocessed_images는 개별 이미지 또는 이미지 배치를 의미합니다.
# 2-2. conv_base.predict(preprocessed_images)는 이 전처리된 이미지를 사전 학습된 VGG19의 컨볼루션 레이어(conv_base)에 통과시켜서 이미지의 특징을 추출합니다.
# 2-3. 이렇게 추출된 features는 원본 이미지 픽셀 정보가 아니라, VGG19가 이미지에서 학습한 고차원적인 패턴(예: 모서리, 질감, 특정 객체 부분 등)을 담고 있는 **특징 맵(feature map)**이 됩니다. 이 특징 맵은 일반적으로 원본 이미지보다 작고 채널 수가 많습니다 (예: 5x5x512).
