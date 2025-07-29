#250725 AM9시. 인라인방식+미세조정(사전학습2+미세조정)
# 인라인 방식의 모델을 완성하고 훈련하는 과정
# 이미 VGG19의 핵심부분(특성추출기)을 불러오셨으니, 추출기 위에 분류기(개/고양이를 구분하는 부분)를 붙이고 전체 모델을 학습시키는 단계
# 인라인 방식: VGG19특성추출기+분류기
# 이미 똑똑한 VGG19의 눈(특성추출기)을 빌려와서, 그 위에 구분할 수 있는 머리(분류기)를 달아주는 것이라고 생각하면 돼요. 
# 이렇게 하나의 몸을 만든 다음, 개와 고양이 사진을 보여주면서 전체 모델이 학습하도록 하는 거죠.

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

# base_path = pathlib.Path("../data/dogs-vs-cats/train")
# new_base_dir = pathlib.Path("../data/dogs-vs-cats/dogs-vs-cats_small")

original_dir = pathlib.Path("../data/dogs-vs-cats/train")
new_base_dir = pathlib.Path("../data/dogs-vs-cats/dogs-vs-cats_small")

# 폴더이동 → 데이터셋은 폴더지정시 폴더이름오름차순정렬해서 자동라벨링.
# dogs-vs-cats_small/
# ├── train/
# │   ├── cats/
# │   │   ├── cat.0.jpg
# │   │   ├── cat.1.jpg
# │   │   └── ... (cat.999.jpg까지)
# │   └── dogs/
# ├── validation/
# └── test/

#이동폴더지정, 시작인덱스, 종료인덱스
def make_subset(subset_name, start_index, end_index): #make_subset("train", 0, 1000)
    for category in ("cat", "dog"):
        dir = new_base_dir/subset_name/category  #windowPath라는 특별한객체, str아님
        os.makedirs(dir, exist_ok=True) #디렉토리가 없을시 새 디렉토리 생성 
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)] 
        for fname in fnames:
            shutil.copyfile(src=original_dir/fname, dst=dir/fname) 

make_subset("train", 0, 1000)
make_subset("validation", 1000, 1500)
make_subset("test", 1500, 2000)

#batch_size에 지정된 만큼 폴더로부터 이미지를 읽어온다. 크기는 image_size에 지정한 값으로 가져온다 
#훈련셋 8:2로 나눠서 검증셋 만드는 방법도 이씨음. subset 특성, seed를 이용해 나눔
# from keras.utils import image_dataset_from_directory 
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

#250725 9시 미세조정 추가
    conv_base.trainable = True
    for layer in conv_base.layers[:4]: #상위4계층만 재훈련
        layer.trainable = False
    #학습률 아주 낮춰서 학습해야함
    #optimizer에 단순 rmsprop하면 안됨

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
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy',
                #   optimizer='rmsprop',
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-5), #-0.00005
                  metrics = ['accuracy'])
    
    #과적합방지(Early Stopping)Keras의 콜백사용해 검증손실이 더이상 개선되지 않을 때 자동학습중단.
    # 테스트셋에서의 손실을 줄이고, 모델의 일반화 성능을 높이는 것이 목표
    #시스템 내부적으로 일처리 완료후, 콜백함수 호출
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="개고양이사전학습2.keras",
            save_best_only=True,
            monitor='val_loss' #검증데이터셋 기준으로 적합한 시기에 호출
        )
    ]

    history = model.fit(train_ds, epochs=50, 
                        validation_data=validation_ds,
                        callbacks = callbacks)
    with open("개와고양이사전학습2.bin", "wb") as file:
        pickle.dump(history.history, file)

if __name__ == "__main__":
    # main()
    deeplearning()
