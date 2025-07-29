# 250718 PM1시
# https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\garbage\garbage
# #폴더명에 공백없어야 데이터가져옴

# 다중 클래스 이미지 분류 (Multi-class classification)
# 입력:	이미지 파일 (.jpg)
# 출력:	클래스 라벨 (0, 1, ..., n)
# 사용모델:	CNN 기반 모델 (Conv2D, MaxPooling2D, Dense 등)
# 라벨링방식:	정수 인코딩 (Label Encoding)
# 목적: 예측 주어진 이미지가 어떤 쓰레기인지 분류

# ✅ MLP란?(Multi-Layer Perceptron)
# = 다층 퍼셉트론
# = 우리가 흔히 말하는 기본형 딥러닝 모델 (Fully Connected Network)
# Dense() 층만으로 구성
# 이미지 → 숫자 벡터로 평탄화 (Flatten)한 후 학습
# Conv2D 나 MaxPooling2D는 없음
# 입출력이 1차원이기 때문에 CNN보다 단순함

# 사진 분류(이진 or 다중) 기초 코드 예제


import os
import imghdr
import numpy as np
import PIL.Image as pilimg

# 1. 데이터 읽기 및 전처리
# 각 클래스(폴더)를 읽어서 64x64로 리사이즈
# 이미지 3채널(RGB)만 사용
# numpy 배열로 저장


# 데이터 경로
base_path = r"C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\garbage\garbage"
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data\garbage\garbage
# batch_size = 32  #한번에 모델에 넣어주는 이미지 개수, 한번에다넣으면 메모리터짐. 너무작으면 느려.
# epochs = 10  #모든데이터이미지 / 배치사이즈 = 에폭스 만큼 반복해서 1에포크 완료-반복해서 똑똑하게만듬

def makeData(garbage_name, label, isTrain=True):   
    if isTrain:
        path = base_path + '/train/' + garbage_name
    else:
        path = base_path + '/test/' + garbage_name


    data = []
    labels = []
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
    return np.array(data), np.array(labels)

# 클래스명 자동 수집
# class_names = sorted(next(os.walk(base_path))[1])
# label_map = {name: idx for idx, name in enumerate(class_names)}
class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

for class_name in class_names:
    path = os.path.join(base_path, class_name)

X, y = [], []
for class_name in class_names:
    data, labels = makeData(class_name, label_map[class_name])
    X.append(data)
    y.append(labels)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
print("전체 데이터 shape:", X.shape, y.shape)

# npz로 저장 후 불러와 사용하면 효율적
# np.savez("garbage_data.npz", data=X, targets=y)





# 🔍 CNN 기반 분류 모델 (기초)
# 목표: 6가지 쓰레기 종류 분류
# 모델: CNN, VGG16, ResNet50, EfficientNet (전이학습 가능)
# 기술 스택: TensorFlow/Keras or PyTorch
# 데이터 전처리: 이미지 크기 조정, 정규화, 증강(Augmentation)
# 성과지표: accuracy, confusion matrix, precision/recall

# 🔍 최소한의 개념(CNN)
# Conv2D	이미지를 훑으면서 특징(모서리, 색 변화 등)을 뽑아냄
# MaxPooling2D	중요한 특징만 남기고 이미지 크기를 줄임
# Flatten	2차원 이미지를 1차원으로 펴서 Dense로 넘김
# Dense	신경망의 fully connected 층
# activation='relu'	음수는 버리고, 양수만 넘김 (속도 빠름)
# activation='softmax'	다중 분류를 위한 출력 확률 계산
# categorical_crossentropy	다중 클래스 분류용 손실 함수
