#250724 AM9시. 인라인방식.
# 인라인 방식의 모델을 완성하고 훈련하는 과정
# 이미 VGG19의 핵심부분(특성추출기)을 불러오셨으니, 추출기 위에 분류기(개/고양이를 구분하는 부분)를 붙이고 전체 모델을 학습시키는 단계
# 인라인 방식: VGG19특성추출기+분류기
# 이미 똑똑한 VGG19의 눈(특성추출기)을 빌려와서, 그 위에 구분할 수 있는 머리(분류기)를 달아주는 것이라고 생각하면 돼요. 
# 이렇게 하나의 몸을 만든 다음, 개와 고양이 사진을 보여주면서 전체 모델이 학습하도록 하는 거죠.

# 1. VGG19의 눈을 고정하기 (동결, Freezing)
# VGG19는 이미 수많은 이미지를 보고 학습해서 사물의 특징을 기가 막히게 잘 찾아냅니다. 
# 예를 들어, 동물의 털 모양, 귀 모양 같은 것들을요. 
# 이런 뛰어난 특징 추출 능력을 망치지 않기 위해, VGG19의 눈 부분은 더 이상 학습하지 못하도록 
# '고정'시켜야 합니다. 이걸 **동결(Freezing)**이라고 해요. 
# 만약 고정하지 않으면, 우리가 가진 몇 안 되는 개/고양이 사진 때문에 
# VGG19가 기존에 배운 것을 잊어버리고 엉뚱하게 변할 수 있습니다.
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

# import gdown #케라스만든사람들이 케라스에 있는 데이터셋 업어오기위해 사용함 
#gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output='dogs-vs-cats.zip')
#안막으면 계속 다운을 받는다 

#복습은 꽃분류로 "../data1/flowers"


base_path = pathlib.Path("../data/dogs-vs-cats/train")
new_base_dir = pathlib.Path("../data/dogs-vs-cats/dogs-vs-cats_small")


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
# │       ├── dog.0.jpg
# │       └── ... (dog.999.jpg까지)
# ├── validation/
# │   ├── cats/
# │   │   ├── cat.1000.jpg
# │   │   └── ... (cat.1499.jpg까지)
# │   └── dogs/
# │       ├── dog.1000.jpg
# │       └── ... (dog.1499.jpg까지)
# └── test/
#     ├── cats/
#     └── dogs/

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
                  optimizer='rmsprop',
                  metrics = ['accuracy'])
    
    #시스템 내부적으로 일처리 완료후, 콜백함수 호출
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="개고양이사전학습2.keras",
            save_best_only=True,
            monitor='val_loss' #검증데이터셋 기준으로 적합한 시기에 호출
        )
    ]

    history = model.fit(train_ds, epochs=10, 
                        validation_data=validation_ds,
                        callbacks = callbacks)
    with open("개와고양이사전학습2.bin", "wb") as file:
        pickle.dump(history.history, file)

if __name__ == "__main__":
    # main()
    deeplearning()


# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning>python 0724개와고양이_사전학습2.py
# 2025-07-24 10:38:25.697969: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-24 10:38:27.804907: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found 2000 files belonging to 2 classes.
# 2025-07-24 10:38:40.383228: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 1000 files belonging to 2 classes.
# Found 1000 files belonging to 2 classes.
# 합성곱 기반층을 동결 전, 훈련가능한 가중치 개수 32
# 합성곱 기반층을 동결 후, 훈련가능한 가중치 개수 0
# 합성곱 기반층을 동결 후, 훈련가능한 가중치 개수 0
# Epoch 1/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 811s 6s/step - accuracy: 0.7566 - loss: 30.8636 - val_accuracy: 0.8900 - val_loss: 5.2502
# Epoch 2/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 747s 6s/step - accuracy: 0.8599 - loss: 6.2869 - val_accuracy: 0.9760 - val_loss: 0.3645
# Epoch 3/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 703s 6s/step - accuracy: 0.8955 - loss: 4.7291 - val_accuracy: 0.9740 - val_loss: 0.5735
# Epoch 4/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 811s 7s/step - accuracy: 0.9004 - loss: 3.1680 - val_accuracy: 0.9760 - val_loss: 0.8201
# Epoch 5/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 784s 6s/step - accuracy: 0.9168 - loss: 3.0036 - val_accuracy: 0.9620 - val_loss: 1.4143
# Epoch 6/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 3319s 27s/step - accuracy: 0.9110 - loss: 3.4048 - val_accuracy: 0.9760 - val_loss: 0.6761
# Epoch 7/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 706s 6s/step - accuracy: 0.9115 - loss: 3.2428 - val_accuracy: 0.9040 - val_loss: 4.7072
# Epoch 8/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 691s 6s/step - accuracy: 0.9105 - loss: 3.0702 - val_accuracy: 0.9440 - val_loss: 1.9873
# Epoch 9/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 712s 6s/step - accuracy: 0.9058 - loss: 2.6156 - val_accuracy: 0.9760 - val_loss: 0.5717
# Epoch 10/10
# 125/125 ━━━━━━━━━━━━━━━━━━━━ 737s 6s/step - accuracy: 0.9183 - loss: 2.7403 - val_accuracy: 0.9790 - val_loss: 0.8845

# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning>python 0723개와고양이_사전학습re.py
# 2025-07-24 14:37:44.085578: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-24 14:37:48.139398: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found 2000 files belonging to 2 classes.
# 2025-07-24 14:37:58.628441: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 1000 files belonging to 2 classes.
# Found 1000 files belonging to 2 classes.
# Traceback (most recent call last):
#   File "C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning\0723개와고양이_사전학습re.py", line 83, in <module>
#     from keras.applications.vhh19 import VGG10
# ModuleNotFoundError: No module named 'keras.applications.vhh19'



# conv_base.summary() #69줄 출력결과
# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\DeepLearning>python 0724개와고양이
# _사전학습2.py
# 2025-07-24 09:20:35.120646: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-07-24 09:20:39.273666: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found 2000 files belonging to 2 classes.
# 2025-07-24 09:21:04.236629: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 1000 files belonging to 2 classes.
# Found 1000 files belonging to 2 classes.
# Model: "vgg19"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)             │ (None, 180, 180, 3)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block1_conv1 (Conv2D)                │ (None, 180, 180, 64)        │           1,792 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block1_conv2 (Conv2D)                │ (None, 180, 180, 64)        │          36,928 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block1_pool (MaxPooling2D)           │ (None, 90, 90, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block2_conv1 (Conv2D)                │ (None, 90, 90, 128)         │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block2_conv2 (Conv2D)                │ (None, 90, 90, 128)         │         147,584 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block2_pool (MaxPooling2D)           │ (None, 45, 45, 128)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv1 (Conv2D)                │ (None, 45, 45, 256)         │         295,168 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv2 (Conv2D)                │ (None, 45, 45, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv3 (Conv2D)                │ (None, 45, 45, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv4 (Conv2D)                │ (None, 45, 45, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_pool (MaxPooling2D)           │ (None, 22, 22, 256)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv1 (Conv2D)                │ (None, 22, 22, 512)         │       1,180,160 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv2 (Conv2D)                │ (None, 22, 22, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv3 (Conv2D)                │ (None, 22, 22, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv4 (Conv2D)                │ (None, 22, 22, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_pool (MaxPooling2D)           │ (None, 11, 11, 512)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv1 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv2 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv3 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv4 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_pool (MaxPooling2D)           │ (None, 5, 5, 512)           │               0 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 20,024,384 (76.39 MB)
#  Trainable params: 20,024,384 (76.39 MB)
#  Non-trainable params: 0 (0.00 B)


# 1.VGG19의 눈을 고정하기 (동결, Freezing) 출력결과
# Found 2000 files belonging to 2 classes.
# Found 1000 files belonging to 2 classes.
# Found 1000 files belonging to 2 classes.
# Model: "vgg19"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer_1 (InputLayer)           │ (None, 180, 180, 3)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block1_conv1 (Conv2D)                │ (None, 180, 180, 64)        │           1,792 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block1_conv2 (Conv2D)                │ (None, 180, 180, 64)        │          36,928 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block1_pool (MaxPooling2D)           │ (None, 90, 90, 64)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block2_conv1 (Conv2D)                │ (None, 90, 90, 128)         │          73,856 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block2_conv2 (Conv2D)                │ (None, 90, 90, 128)         │         147,584 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block2_pool (MaxPooling2D)           │ (None, 45, 45, 128)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv1 (Conv2D)                │ (None, 45, 45, 256)         │         295,168 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv2 (Conv2D)                │ (None, 45, 45, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv3 (Conv2D)                │ (None, 45, 45, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_conv4 (Conv2D)                │ (None, 45, 45, 256)         │         590,080 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block3_pool (MaxPooling2D)           │ (None, 22, 22, 256)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv1 (Conv2D)                │ (None, 22, 22, 512)         │       1,180,160 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv2 (Conv2D)                │ (None, 22, 22, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv3 (Conv2D)                │ (None, 22, 22, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_conv4 (Conv2D)                │ (None, 22, 22, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block4_pool (MaxPooling2D)           │ (None, 11, 11, 512)         │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv1 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv2 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv3 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_conv4 (Conv2D)                │ (None, 11, 11, 512)         │       2,359,808 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ block5_pool (MaxPooling2D)           │ (None, 5, 5, 512)           │               0 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 20,024,384 (76.39 MB)
#  Trainable params: 20,024,384 (76.39 MB)
#  Non-trainable params: 0 (0.00 B)
# 합성곱기반층을 동결 전, 훈련가능한 가중치 개수 32
# 합성곱기반층을 동결 후, 훈련가능한 가중치 개수 0
