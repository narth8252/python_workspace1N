#250723 PM1시-CNN 쌤PPT415p (250717딥러닝종합_백현숙)
#0722개와고양이분류.py코드보다 간결.
# 투스테이지 : 사전학습된 모델로(미세조정,특성추출 등) 개와 고양이 이미지를 분류하는 과정
#1. 특성추출하기 Feature Extraction - CNN하고 완전피드포워딩
#2. 사전(Pre-trained Model)학습된 모델을 불러와서 CNN파트랑 완전연결망을 쪼개서
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
# VGG19, ResNet, MobileLet 등 이미지셋(학습된) 모델들이 있다.
# VGG-19 Tensorflow구현 참고 https://hwanny-yy.tistory.com/11 
# (base) C:\Windows\System32>conda activate mytensorflow
# (mytensorflow) C:\Windows\System32>conda install gdown
import os, shutil, pathlib
import tensorflow as tf
import keras
from keras.utils import image_dataset_from_directory

original_dir = pathlib.Path("../data/dogs-vs-cats/train")
new_base_dir = pathlib.Path("../data/dogs-vs-cats/dogs-vs-cats_small")
#특성추출 Feature Extraction 방식 4단계
# 1.파일복사나 이동(재료준비) → 2.특성저장(핵심재료추출) → 3.분류모델학습(요리) → 4.예측(맛보기)

# 1.복사이동폴더지정, 시작인덱스, 종료인덱스
def copyImage():
    def make_subset(subset_name, start_index, end_index):
    #make_subset("train", 0, 1000)
        for category in ("cat", "dog"):
            dir = new_base_dir/subset_name/category #windowPath라는 특별한객체,str아님
            os.makedirs(dir, exist_ok=True) #dir없을시 new생성
            fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index, )]
            for fname in fnames:
                shutil.copyfile(src=original_dir/fname, dst=dir/fname)
    # 작은 데이터셋 폴더구조 만들기: 원본이미지를 다사용하기는 부담스러우니, dogs-vs-cats_small이라는 새폴더 안에 훈련(train), 검증(validation), 테스트(test) 로 이미지를 나누어 복사
    # 자동 라벨링을 위한 폴더 정리
    make_subset("train", 0, 1000)  #학습용1000장(cat,dog각각)
    make_subset("validation", 1000, 1500) #검증셋 각500장:학습도중 성능확인,과대적합(overfitting)여부확인에 사용
    make_subset("test", 1500, 2000) #테스트셋 각500장:모델학습종료후, 최종성능 평가에 사용

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
# 2.특성저장(핵심재료추출):save_features() 함수와 get_features_and_labels() 함수
# • conv_base: VGG19모델 中 이미지특성추출하는 합성곱기반층(Convolutional Base)만 가져옴(include_top=False옵션).
#              이 부분은 이미 너무 학습 잘돼있어서, 내 데이터로 재학습하면 오히려 성능저하 될수있어서,
#              이 부분은 동결Freezing시켜 더이상 학습되지 않도록.
# • 특성추출과정(get_features_and_labels):
#   - `train_ds, validation_ds, test_ds`에서 batch(묶음)단위로 이미지 가져온다
#   - 가져온 이미지를 VGG19가 이해할수있는 형태로 전처리`preprocess_input`(예:픽셀값 범위조정 등)
#   - 전처리된 이미지를 `conv_base`에 넣으면 VGG19가 이 이미지에서 뽑아낸 `핵심숫자특징Feature`이 나옴.
#     이 Feature들은 원본이미지보다 훨씬 작지만, 본질적인 정보를 담음(VGG19dml `block5_pool` 레이어에서 나오는 `(5,5,512)`크기의 특징맵   
#   - 이렇게 추출된 특징들과 해당이미지의 실제라벨을 모두 모아서 `all_geatures`와 `all_labels`리스트에 저장
# • 특성저장`save_features`: 추출된 모든 특징데이터(train, validation, test)를 `개와고양이_특성.bin`이라는 파일로 저장
#   - 왜 저장? VGG19로 특징추출하는 과정은 시간이 오래걸림. 한번추출해서 저장해놓고 모델 여러번 학습시킬때 이 시간을 절약하기위해.
#           이미 추출된 `핵심재료`를 바로 불러와서 사용하기위해. 이 특징데이터는 Numpy배열형태임.
#VGG19 모델 가져오기
from keras.applications.vhh19 import VGG10
conv_base = keras.applications.vgg19.VGG19(
    weights="imagenet",
    include_top=False, #CNN만 가져오기, 하단에 있음
    input_shape=(180, 180, 3) #데이터셋ds에서 정한크기와 일치하게 입력할 데이터크기 주기
)

conv_base.summary() #CNN요약
# block5_pool (MaxPooling2D)  (None, 5, 5, 512)  0 

# 데이터셋을 주고 CNN으로부터 특성추출해서 전달하는 함수
import matplotlib.pyplot as plt
import numpy as np
def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg19.preprocess_input(image)
        print(images.shape, preprocessed_images.shape)
        # plt.show()
        # break
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

import pickle
def save_features():
    train_features, train_labels = get_features_and_labels(train_ds)
    validation_features, validation_labels = get_features_and_labels(validation_ds)
    test_features, test_labels = get_features_and_labels(test_ds)

    # • 히스토리저장 : 훈련학습과정에서 모델의 손실loss과 정확도accuracy변화를 기록한 history객체를 `dog_vs_cat.bin`으로 저장.
    #               이 데이터는 학습곡선그래프를 그릴때 유용
    features = {"train_features":train_features, "train_labels":train_labels,
                "validation_features":validation_features, "validation_labels":validation_labels,
                "test_features":test_features, "test_labels":test_labels}
    try:
        with open("개와고양이_특성.bin", 'wb') as file:
            pickle.dump(features, file)
        print("개와고양이_특성 저장완료")
    except Exception as e:
        print(f"개와고양이_특성 저장중 오류발생 {e}")

def load_features():
    try:
        with open("개와고양이_특성.bin", 'rb') as file:
            features = pickle.load(file)
            print("개와고양이_특성 로딩 성공")
    except Exception as e:
        print(f"개와고양이_특성 로딩중 실패 : {e}")

    return features

# 3.요리하기  `deeplearning()`함수
# • 특성불러오기: `load_features()`함수를 통해 저장했던 `개와고양이_특성.bin`에서 학습에 필요한 특징데이터 불러오기
# • 데이터증강`data_argumentation`:데이터증강을 적용해 더 다양한 특징을 보고 학습하도록 돕기
#                                '인라인방식'에서는 원본이미지에 적용했었음
# • 분류기(머리) 만들기
#   - `inputs=keras.Input(shape=(5,5,512))`: VGG19의 마지막합성곱레이어에서 추출된 특성크기인(5,5,512)를 모델의 입력으로 받기
#   - `Flatten():`(5,5,512)`형태의 2D특성맵을 1D벡터로 쫙 펼쳐줌
#   - `Dense`layers: 펼쳐진 벡터를 입력으로 받아 이미지를 분류하는 '완전연결층 Fully Connected Layer, 즉 분류기'를 만든다.
#                   이 레이어들이 "이 특징들을 보니 개같아! 고양이같아!"라고 판단하는 역할
#   - `Dropout(0.5)`: 과대적합 막기위해 일부뉴런을 무작위로 비활성화하는 드롭아웃 레이어를 추가
#   - `Dense(1,activation='sigmoid')`: 최종출력 레이어.
#            이진분류(개/고양이)이므로 뉴런1개에 sigmoid활성화 함수를 사용. 시그모이드는 0과1사이 확률값 출력.
# • 모델 컴파일: 모델이 학습시작전 어떤 손실함수 loss function을 사용할지, 어떤 최적화도구 optimizer를 사용할지, 어떤 평가지표metrics를 볼지 설정
#   -`loss='binary_crossentropy`: 이진분류에 적합한 손실함수
#   -`optimizer='rmsprop`:학습을 효율적으로 진행시키는 최적화 알고리즘
#   -`metrics=['accuracy']`:모델정확도를 평가지표로 사용하겠다
# • 모델학습 `model.fit`
#   -`epochs=30`:전체 훈련셋30번 반복학습
#   -`validation_data`: 학습도중 검증셋으로 모델성능 확인
#   -`callbacks`(ModelCheckPoint): 훈련중 가장 좋은성능(이 코딩에서는 `val_loss`가장낮을때)을 보인 모델의 
#                가중치를 자동으로 `특성추출.keras`파일로 저장. 이러면 나중엥 최고성능모델사용가능
from keras import models, layers
def deeplearning():
    #특성추출, 불러오기, 예측
    features = load_features()

    data_argumention = keras.Sequential([
        layers.RandomFlip("horizontal"), #이미지를 수평으로 무작위로 뒤집기
        layers.RandomRotation(0.1),      #이미지를 최대2pi라디언의 10(36도)까지 무작위회전
        layers.RandomZoom(0.1)           #이미지를 최대10%까지 무작위 확대하거나 축소
    ])

    #맨마지막 block
    inputs = keras.Input(shape=(5,5,512))
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(128)(s)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')
    model = keras.Model(inputs, outputs)
    #과소과대 무관하게 학습완료해야 저장되는데, 과대적합되는 시점에 저장가능
    #모델에 학습도중 과대적합되는 걸 확인가능
    #콜백함수에 저장할 파일명을 전달하면 자동호출해서 list형태로 받아감
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="특성추출.keras",
            save_best_only=True,    #가장 적합할때 저장
            monitor="val_loss"      #검증데이터의 손실loss값이 최적화일때
        )
    ]
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    history = model.fit(features["train_features"], features["train_labels"],
                        epochs=30,
                        validation_data=(features["validation_features"], features["validation_labels"]),
                        callbacks=callbacks)
    
    #모델 저장
    try:
        model.save("dog_vs_cat.keras")
        print("모델 저장 완료")
    except Exception as e:
        print(f"모델 저장 중 오류 발생 {e}")

    #히스토리 저장
    try:
        with open("dog_vs_cat.bin", 'wb') as file:
            pickle.dump(history.history, file)
        print("히스토리 저장 완료")
    except Exception as e:
        print(f"히스토리 저장 중 오류 발생 {e}")

# 4. 예측: predict() 함수
# • 학습된 모델 불러오기: 특성추출.keras 파일에 저장된 최적의 모델 가중치를 불러옵니다.
# • 테스트 특성 불러오기: load_features()를 통해 테스트셋의 특징(test_features)과 실제 라벨(test_labels)을 가져옵니다.
# • 예측 수행: 불러온 모델에 테스트셋의 특징을 입력하여 예측 값(predictions_batch)을 얻습니다. 이 값은 0에서 1 사이의 확률 값입니다.
# • 클래스 분류: 예측된 확률 값이 0.5보다 크면 '개'(1), 작으면 '고양이'(0)로 최종 분류합니다.
# • 결과 출력: 전체 테스트 이미지 수 대비 맞춘 개수와 틀린 개수를 출력하여 모델의 성능을 간단하게 확인합니다.
def predict():
    #학습된 모듈 불러오기
    loaded_model_keras = None
    try:
        loaded_model_keras = keras.models.load_model("특성추출.keras")
        print("모델 부르기 성공")
    except Exception as e:
        print(f"모델 로딩중 실패: {e}")
        return

    features = load_features()
    test_features, test_labels = features["test_features"], features["test_labels"]

    # val_ds가 폴더로부터 이미지파일 읽어오는데 batch_size만큼씩 읽어온다
    total_samples_proceed = len(test_labels)

    #예측하기
    predictions_batch = loaded_model_keras.predict(test_features, verbose=2)
    predicted_class = (predictions_batch>0.5).astype(int).flatten() # 0.5보다 큰 거는 True, 작은 거는 False

    total_match_count = np.sum(predicted_class == test_labels)

    # print(total_samples_proceed, len(labels_batch))
    print("전체 데이터 개수: ", total_samples_proceed)
    print("맞춘 개수: ", total_match_count)
    print("틀린 개수: ", total_samples_proceed - total_match_count)

def main():
    while True:
        print("1. 파일복사")
        print("2. 특성저장")
        print("3. 학습")
        print("4. 예측")
        sel = input("선택 >> ")
        if sel == "1":
            copyImage()
        elif sel == "2":
            save_features()
        elif sel == "3":
            deeplearning()
        elif sel == "4":
            predict()
        else:
            break

if __name__ == "__main__":
    main()
#🔶이 방식의 장점 다시 정리
# • 데이터셋이 적을 경우: 우리가 가진 개/고양이 사진이 수백, 수천 장밖에 안 될 때, 이미 수백만 장으로 학습된 VGG19의 강력한 특징 추출 능력을 활용하면 적은 데이터로도 좋은 성능을 낼 수 있습니다.
# • 학습 시간을 줄여줌: VGG19의 합성곱 기반층은 동결된 채로 한 번만 특징을 추출하여 저장해두면, 분류기만 학습시키기 때문에 전체 학습 시간이 훨씬 짧아집니다.
# • 컴퓨터 자원이 적어도 가능: 대규모 CNN 모델을 처음부터 학습시키는 것은 매우 큰 컴퓨팅 자원을 필요로 하지만, 이미 학습된 모델의 특징 추출 부분은 그대로 가져다 쓰고 분류기만 학습하므로 비교적 적은 자원으로도 가능합니다.












































# 2.특성저장(핵심재료추출):save_features() 함수와 get_features_and_labels() 함수
#🔶VGG19라는 똑똑한 눈을 빌려와서 핵심특성만 뽑아내는 과정
# • VGG19? 이미 수백만장의 다양한 이미지(ImageNet)를 학습해 특성을 기가막히게 찾아내는 사전훈련된 대규모CNN모델.
#   "이게 고양이다"라고 직접 말하기보다는,"이사진엔 털이있고, 뾰족한귀가있고, 눈이 동그랗네"처럼 이미지의 중요한 시각적특징Feature들을 추출
# 🔶합성곱(Convolution나선형): 두 함수를 "겹쳐서 곱하고 더하는" 연산. 한 함수가 다른 함수 위를 "훑고 지나가면서" 특정패턴을 찾아내는 과정. 
#        이미지 처리시 작은필터(커널)가 이미지위를 움직이면서 픽셀값들을 곱하고더해서 새로운값을 만들어내는 방식으로 사용.
#
#🔶Convolutional Base 합성곱 기반층
# 합성곱신경망(CNN)에서 이미지로부터 유의미한 특징(Feature)을 추출하는 부분
# • 역할: CNN은 여러층의 Convolution layers와 Pooling layers로 구성.
#       이 레이어들은 이미지의 가장 기본적특성(예:선,모서리,색상대비)부터 시작해서 점점더 복잡한 특징(예:질감,패턴,객체의 부분)을 계층적Dense으로 추출함.
#       Conv_base는 이 특징추출을 담당하는 신경망의 핵심부분
# • Base인 이유: 보통 대규모 이미지데이터셋(예:ImageNet)으로 미리학습된 CNN모델(VGG,ResNet, MobileNet 등)을 가져와 사용할때,
#               모델의 가장 아래부터 특징추출get_features을 담당하는 합성곱층까지만을 Base라고 함.
#               이미 학습했기때문에 매우강력
# • Top과의 대비: CNN모델의 맨위에는 보통 추출된 특징들을 바탕으로 최종적으로 이미지를 분류하는 완전연결층Fully Connected Layer 또는 분류기Classifier Head가 붙어있음.
#               이 부분을 Top이라고도함. Conv_base는 Top부분을 제외한 특징추출기라고 보면됨
# • 전이학습 Transfer Learning 활용: 개고양이분류처럼 특정목적의 작은 데이터셋을 가지고있을때, 처음부터 거대한CNN모델을 학습시키는 대신,
#           이미 ImageNet등으로 학습된 VGG19같은 모델의 Conv_base를 가져와 사용.
#           눈은 똑똑하니 그대로 쓰고, 그위에 목적에 맞는 새로운 머리(분류기)만 붙여서 학습시키는 방식으로 하면,
#           적은 데이터로도 빠르고 높은 성능 얻을수 있음

#🔶합성곱신경망(CNN) Convolutional Neural Network
# 이미지나 영상과 같은 시각적 데이터를 분석하고 처리하는 데 특화된 인공 신경망의 한 종류
# 기존의 일반적인신경망(완전연결신경망)은 이미지를 1차원데이터로 펼쳐서 입력받으므로, 이미지의 중요한 공간정보(픽셀간 위치관계)가 손실됨
# 이를 해결하기 위해 합성곱 Convolition이라는 연산을 사용
# • 합성곱연산: 이미지의 작은영영을 filter 또는 kernel이라는 작은 행렬로 훑으며 이미지의 feature(예:선,모서리,색상,질감 등)추출,
#           마치 사람의 눈으로 자세히 들여다보며 파악하는것처럼 이미지전체를 이동하며 적용되기때문에
#           특징이 이미지의 어느위치에 있든 동일하게 감지할 수 있다.

#🔶CNN의 주요특징
# • 공간적 정보유지: 이미지를 1차원으로 펼치지않고 2차원(또는 3차원) 그대로 처리해 픽셀간의 공간적 관계 보존
# • 특징자동추출: 직접 이미지특징을 설계할필요없이, 네트워크가 학습과정에서 이미지의 유용한 특징filter을 자동으로 찾아냄
# • 계층적특징학습: 초기레이어에서는 간단한특징(선,모서리)을 학습하고, 깊은레이어로 갈수록 이 간단한 특징들을 조합해 더 복잡하고 추상적인 특징(눈,코,귀, 객체의 부분)을 학습
# • 매개변수공유: 동일한 필터를 이미지의 여러위치에 적용해 학습해야할 매개변수(가중치)의 수를 크게 줄여줘, 
#               모델의 효율성을 높이고 과대적합을 줄임