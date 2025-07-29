#250724 PM3시. 인라인방식. 쌤 꽃분류5.py

"""
1. 투스테이지(개와고양이_사전학습1.py)
CNN동결 VGG19의 특징을 미리 계산하고 numpy배열로 바꾼다 저장된 특성으로 분류학습을 다시한다
장점 : 훈련속도가 빠르다
단점 : 메모리를 많이 차지한다
학습 데이터셋 커지면 힘들다, 데이터증강 적용방식이 내 데이터가 아니라 추출한 특성에 적용된다.

2. 인라인(개와고양이_사전학습2.py)
VGG19특성 추출부분을 전체 모델안에 포함시킨다. 분류학습을 한다
장점 : 원본이미지에 데이터 증강이 바로 적용. 과대적합 방지.
속도는 투스테이지보다 느리다.
보통 많이 쓰는 방법이다.
 """

#VGG19 
import os, shutil, pathlib 
import keras.callbacks
import tensorflow as tf 
import keras
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory 
from keras.applications.vgg19 import VGG19
from keras import models, layers 

#복습은 꽃분류로 "../data1/flowers"
original_dir = pathlib.Path("../data1/flowers")
new_base_dir = pathlib.Path("../data1/flowers/flowers_small")

#폴더지정, 시작인덱스, 종료인덱스
def make_subset(subset_name, startIndex=0, endIndex=700): #make_subset("train", 0, 1000)
    for category in ("daisy", "dandelion", "tulip", "rose", "sunflower"):
        dir = new_base_dir/subset_name/category #WindowsPath 라는 특별한 객체, str 아님
        os.makedirs(dir, exist_ok=True) #디렉토리가 없을 경우 새로 디렉토리를 만들어라
        #파일명 cats0.jpg cats1.jpg
        dataList = os.listdir(original_dir/category) #리스트를 가져와서
        if endIndex != -1:
            fnames = dataList[startIndex:endIndex]
        else: #데이터개수가 몇개인지 몰라서 endIndex값이 -1이 오면
            fnames = dataList[startIndex: ]
        for fname in fnames:
            shutil.copyfile(src=original_dir/category/fname, dst=dir/fname)     

#데이터가 적어서700건까지만 훈련셋하고 validation검증셋 안함
make_subset("train", 0, 700)
# make_subset("validation", 1000, 1500)
make_subset("test", 700, -1)

# from keras.utils import image_dataset_from_directory 
#데이터가 많지않아서 검증validation폴더를 따로 쪼갤수 없을때
train_ds = image_dataset_from_directory(
    new_base_dir/"train", 
    seed=1234,            #seed: 동일하게 자르게 하기위해
    subset='training',    #훈련셋
    validation_split=0.2, #얼마만큼의 비율로 자를거냐
    image_size=(180,180),
    batch_size=16
)
validation_ds = image_dataset_from_directory(
    new_base_dir/"train", 
    seed=1234,            
    subset='validation',
    validation_split=0.2,
    image_size=(180,180),
    batch_size=16
)
test_ds = image_dataset_from_directory(
    new_base_dir/"test", 
    image_size=(180,180),
    batch_size=16
)

from keras import models, layers
def deeplearning():
    #vgg19 이미지 모델 가져오기 
    # from keras.applications.vgg19 import VGG19 
    conv_base = keras.applications.vgg19.VGG19(
        weights="imagenet", 
        include_top=False, #CNN만 가져와라 , CNN이 하단에 있음, 상단-완전연결망(분류)
        input_shape=(180, 180, 3) #입력할 데이터 크기(180,180), 색정보(3) 줘야함 
        #데이터셋에서 지정한 크기와 일치해야 한다 
    ) 
    # conv_base.summary() #CNN요약, 보고 주석처리해도됨

    # 1.VGG19의 눈을 고정하기 (동결, Freezing)
    conv_base.trainable = True
    print("합성곱 기반층을 동결 전, 훈련가능한 가중치 개수", len(conv_base.trainable_weights))
    conv_base.trainable = False #동결
    print("합성곱 기반층을 동결 후, 훈련가능한 가중치 개수", len(conv_base.trainable_weights))
    # 합성곱 기반층을 동결 전, 훈련가능한 가중치 개수 32
    # 합성곱 기반층을 동결 후, 훈련가능한 가중치 개수 0

    # 2. 분류기(머리) 만들기 및 붙이기
    #데이터 증강에 필요한 파라미터들
    data_argumetation = keras.Sequential( [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.4)
    ])

    #모델만들기
    inputs = keras.Input(shape=(180,180,3)) #모델의 입력레이어 정의
    x = data_argumetation(inputs)    #입력이미지에 데이터증강적용
    x = keras.applications.vgg19.preprocess_input(x) #필수:vgg19에 맞는 전처리(픽셀값범위 조정 등)

    #인라인 방식으로 CNN연결
    x = conv_base(x) #특성추출과정(오래걸림)
######################################################
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(64)(x)
    outputs = layers.Dense(5, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics = ['accuracy'])
    
    #시스템 내부적으로 일처리 완료후, 콜백함수 호출
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="꽃분류사전학습.keras",
            save_best_only=True,
            monitor='val_loss' #검증데이터셋 기준으로 적합한 시기에 호출
        )
    ]

    history = model.fit(train_ds, epochs=10, 
                        validation_data=validation_ds,
                        callbacks = callbacks)
    with open("꽃분류사전학습.bin", "wb") as file:
        pickle.dump(history.history, file)

if __name__ == "__main__":
    # main()
    deeplearning()
