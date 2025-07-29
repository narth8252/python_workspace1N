#250722 PM2시. 꽃이미지분류를 위한 컨볼루션신경망(CNN)을 훈련하고 저장
import keras.utils
from tensorflow.keras.models import load_model # load_model 사용을 위해 필요
from keras import models, layers
import matplotlib.pyplot as plt # 차트 그리기 위해 필요
import tensorflow as tf
import numpy as np
import pickle # history 저장/로드를 위해 필요
import keras
import os
import shutil # ImageCopy 함수에서 사용됨

# 2. study() 함수 정의: 모든 핵심 로직을 코드구조화
def study():
    print("--- \n딥러닝 모델 생성 및 학습 시작 ---")
    img_height = 180  # 모든이미지의 조정 높이
    img_width = 180   # 모든이미지의 조정 너비
    batch_size = 32 #훈련 중 함께 처리될 샘플(이미지)의 개수

    # 3. 설정 및 데이터 증강 정의
    #데이터 증강은 기존 이미지에 무작위적이지만 현실적인 변환을 적용하여 훈련 데이터의 다양성을 늘리는 기술입니다. 예를 들어, 꽃 이미지를 수평으로 뒤집어도 그 꽃의 종류는 변하지 않지만, 모델에게는 동일한 객체의 "새로운" 관점을 제공합니다.
    #과적합(overfitting)방지. 과적합은 모델이 훈련 데이터를 너무 잘 학습하여 (노이즈와 특정 특징까지 포함하여) 새로운, 보지 못한 데이터에서는 성능이 떨어지는 현상입니다. 데이터를 증강함으로써 모델은 더 다양한 입력들을 보게 되어 더욱 견고해지고 실제 이미지에 더 잘 일반화됩니다.
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"), # 이미지를 수평으로 무작위 뒤집기
        layers.RandomRotation(0.1),  # 이미지를 무작위로 최대 36도 (360도의 10%) 회전
        layers.RandomZoom(0.1)     # 이미지를 무작위로 최대 10% 확대/축소
    ]
)
    
    # 4. 모델 아키텍처 (CNN 정의)
    # Keras의 Sequential API를 사용하여 컨볼루션 신경망(CNN) 모델을 정의합니다. Sequential 모델은 신경망을 선형 스택으로 레이어별로 구축합니다.
    model = models.Sequential()
    model.add(data_augmentation) # 데이터 증강 레이어 추가 (다른 처리보다 *먼저* 변환 적용)
    #과대적합을 막기 위해, 증강에 대한 파라미터를 주면, 1에포크마다 데이터를 조금씩 변형을 가해서 가져간다
    model.add(layers.Rescaling(1./255)) # 픽셀 값을 0-255 범위에서 0-1 범위로 정규화

    #컨볼루션 기본
    model.add(layers.Conv2D(32, (3,3), activation='relu')) # 첫 번째 컨볼루션 레이어: 32개 필터, 3x3 커널, ReLU 활성화
    model.add(layers.MaxPooling2D((2,2))) #최대풀링: 2x2 윈도우에서 최대값을 취하여 특징 맵 크기 축소
    model.add(layers.Conv2D(64, (3,3), activation='relu')) #두번째 컨볼루션 레이어: 64개 필터, 3x3 커널, ReLU 활성화
    model.add(layers.MaxPooling2D((2,2)))

    # 분류 헤드 (완전 연결 레이어)
    model.add(layers. Flatten()) #CNN과 완전연결망 연결, 2D특징맵을 1D벡터로 평탄화
    model.add(layers.Dropout(0.3)) # 드롭아웃: 훈련 중 입력 단위의 30%를 무작위로 0으로 설정하여 과적합 방지
    model.add(layers.Dense(256, activation='relu')) # 첫 번째 완전 연결 레이어 (Dense): 256개 유닛, ReLU 활성화
    model.add(layers.Dense(128, activation='relu')) # 두 번째 완전 연결 레이어 (Dense): 128개 유닛, ReLU 활성화
    model.add(layers.Dense(5, activation='softmax')) # 출력 레이어: 5개 유닛 (5개 꽃 클래스), 다중 클래스 확률 분포를 위한 Softmax 활성화

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', #라벨-원핫인코딩안하려고
                  metrics=['accuracy'])
    
    print("--- 모델 아키텍처: ----\n")
# C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\data1\flowers
    train_dir = "../data1/flowers"  # 각 꽃 클래스에 대한 하위 폴더가 포함된 루트 디렉토리
    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2, # 데이터의 20%를 검증용으로 분할
        subset='training',    # 이 데이터셋이 훈련 부분을 포함하도록 지정
        seed=1234,            # 일관된 분할을 보장 (매번 동일한 이미지가 훈련/검증으로 할당됨)
        image_size=(img_height, img_width),  # 모든 이미지를 180x180 픽셀로 크기 조정
        batch_size=batch_size  # 이미지를 32개씩 배치로 구성
        )
    
    val_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='validation',  # 이 데이터셋이 검증 부분을 포함하도록 지정
        seed=1234,            # **중요**: 겹치지 않는 분할을 위해 훈련셋과 동일한 시드를 사용
        image_size=(img_height, img_width),
        batch_size=batch_size
        )
    
    # 7. 모델 훈련
    history = model.fit(train_ds,
                        epochs=30,
                        validation_data = val_ds)
    
    # 8. 모델 및 기록 저장
    model.save("flowers_model.keras")
    f = open("flowers_hist.hist", "wb")
    pickle.dump(history.history, file=f) #history만 저장시 에러, .history확장자추가
    f.close()
    # model.save("flowers_model.keras")
    # with open("flowers_hist.pkl", "wb") as f: # 확장자를 .pkl로 변경하는 것이 일반적
    #     pickle.dump(history.history, f)
    # print("\n딥러닝 모델 훈련 및 저장 완료! 'flowers_model.keras'와 'flowers_hist.history' 파일로 저장되었습니다. ✨")

#히스토리를 읽어서 차트 그리기
def drawChart():
    print("--- \n훈련 기록을 읽어 차트 그리기 시작 ---")
    f = open("flowers_hist.pkl", "rb")
    history = pickle.load(f)
    f.close()
    print("히스토리 키:", history.keys())

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    #bo : 파란원으로 그리겠다.
    plt.figure(figsize=(12, 6)) # 그래프 크기 설정
    plt.figure() #새창으로 팝업
    plt.plot(epochs, acc, 'bo', label="Training acc")
    plt.plot(epochs, val_acc, 'b', label="Validation acc")
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()

#예측
def predict():
    print("--- \n모델 로드 및 예측 시작... ---")
    #1.keras 내 모델가져오기 "flowers_model.keras"
    model = load_model("flowers_model.keras")
    # \Data_Analysis_2507\data1\flowers\test
    test_dir="../data1/flowers/test"
    test_ds = keras.utils.image_dataset_from_directory( # keras.preprocessing -> keras.utils
        test_dir,
        seed=1234,
        image_size=(180, 180), # 모델 훈련 시 사용한 이미지 크기와 동일하게 설정
        batch_size=1, #1개씩 가져오기
        shuffle=False # 예측 시에는 데이터 순서를 섞지 않는 것이 좋습니다.
    )
    print(f"테스트 데이터셋 클래스 이름: {test_ds.class_names}")
    print(f"테스트 데이터셋에 약 {tf.data.experimental.cardinality(test_ds).numpy() * batch_size}개의 이미지가 있습니다.")

    i=0
    match_cnt = 0
    for images, labels in test_ds:
        #테스트데이터셋으로부터 이미지들과 라벨즈를 갖고 온다
        output=model.predict(images)
        print("라벨 : ", labels, output)
        for i in range(0, len(labels)):
            if labels[i] == np.argmax(output[i]): #softmax 라벨이 확률로 온다, 그중에 가장 큰값의 인덱스
                match_cnt+=1
    print("일치 개수: ", match_cnt)
    print("불일치개수: ", tf.data.Dataset.cardinality(test_ds).numpy()- match_cnt)

def main():
    while (True):
        print("\n--- 메뉴 ---")
        print("1. 기본 학습 (모델 훈련)")
        print("2. 차트 그리기 (훈련 기록 시각화)")
        print("3. 예측하기 (테스트 데이터로 예측)")
        print("4. 평가하기 (테스트 데이터로 모델 성능 평가)")
        print("------------")

        sel = input("선택 : ")
        if sel == "1":
            study()
        elif sel=="2":
            drawChart()
        else:
            return

# --- 프로그램 시작점 ---   
if __name__=="__main__":
    # ImageCopy() # 데이터셋 구성이 아직 안 되어있다면 이 줄의 주석을 풀고 한 번 실행하세요.
                  # 이미 되어있다면 주석 처리된 상태를 유지합니다.
    # study() # 훈련이 완료되었거나, 새로 훈련하고 싶을 때만 이 줄의 주석을 풀고 실행합니다.
              # 현재 훈련 중이시라면 이 줄은 계속 주석 처리된 상태로 두세요.
    # drawChart() # 훈련 완료 후 차트를 바로 보고 싶다면 사용
    # main() # 이 줄을 실행하여 메뉴 시스템을 시작합니다.

# 3.설정 및 데이터 증강 정의
# study() 함수 내부에서는 이미지 처리 및 훈련을 위한 몇 가지 주요 매개변수를 정의합니다:
# • img_height = 180: 모든 이미지의 높이를 조정할 목표 크기입니다.
# • img_width = 180: 모든 이미지의 너비를 조정할 목표 크기입니다.
# • batch_size = 32: 훈련 중 한 번의 순전파/역전파 과정에서 함께 처리될 샘플(이미지)의 개수입니다. 배치 처리는 효율적인 계산과 안정적인 기울기 업데이트에 도움이 됩니다.

# 그 다음, data_augmentation 레이어를 정의합니다:
# • 데이터 증강이란? 💡 데이터 증강은 기존 이미지에 무작위적이지만 현실적인 변환을 적용하여 훈련 데이터의 다양성을 늘리는 기술입니다. 예를 들어, 꽃 이미지를 수평으로 뒤집어도 그 꽃의 종류는 변하지 않지만, 모델에게는 동일한 객체의 "새로운" 관점을 제공합니다.
# • 왜 사용하나요? 과적합(overfitting)을 방지하는 데 도움이 됩니다. 과적합은 모델이 훈련 데이터를 너무 잘 학습하여 (노이즈와 특정 특징까지 포함하여) 새로운, 보지 못한 데이터에서는 성능이 떨어지는 현상입니다. 데이터를 증강함으로써 모델은 더 다양한 입력들을 보게 되어 더욱 견고해지고 실제 이미지에 더 잘 일반화됩니다.

# 4. 모델 아키텍처 (CNN 정의)
# 이 섹션에서는 Keras의 Sequential API를 사용하여 컨볼루션 신경망(CNN) 모델을 정의합니다. Sequential 모델은 신경망을 선형 스택으로 레이어별로 구축합니다.
# •  model.add(data_augmentation): 이미지가 통과하는 첫 번째 "레이어"입니다. 스케일링 전에 이미지가 무작위로 변환됩니다.
# • layers.Rescaling(1./255): 픽셀 값을 원본 범위(0-255)에서 0-1 범위로 스케일링하는 레이어입니다. 이 정규화는 신경망이 더 작고 표준화된 입력 값으로 더 빠르고 안정적으로 훈련되도록 돕습니다.
# • layers.Conv2D: 컨볼루션 레이어입니다. 필터(커널)를 적용하여 이미지에서 패턴(특징)을 학습합니다. 예를 들어, 한 필터는 가장자리를 감지하고 다른 필터는 텍스처를 감지할 수 있습니다. 32와 64는 사용되는 필터의 수를 나타냅니다. (3,3)은 필터의 크기입니다. activation='relu'는 비선형성을 도입하여 모델이 더 복잡한 관계를 학습할 수 있도록 하는 ReLU (Rectified Linear Unit) 활성화 함수입니다.
# • layers.MaxPooling2D: 풀링 레이어입니다. 특징 맵의 공간적 차원(너비와 높이)을 줄여 다음과 같은 이점을 제공합니다:
# ㆍ 매개변수 수를 줄여 모델을 더 빠르게 만듭니다.
# ㆍ 이미지의 약간의 이동(translation invariance)에 대해 모델을 더 견고하게 만듭니다.
# ㆍ (2,2)는 각 2x2 윈도우에서 최대값을 취하는 것을 의미합니다.
# • layers.Flatten(): 컨볼루션 및 풀링 레이어가 특징을 추출한 후, Flatten()은 2D 특징 맵을 1D 벡터로 변환합니다. 이는 이후의 완전 연결(Dense) 레이어가 1D 입력을 기대하기 때문에 필요합니다.
# • layers.Dropout(0.3): 정규화 기법입니다. 훈련 중에 이전 레이어의 뉴런 중 30%를 무작위로 "끔"으로써 복잡한 훈련 데이터에 대한 과도한 적응을 방지하고, 네트워크가 더 견고한 특징을 학습하도록 유도하여 과적합을 줄입니다.
# • layers.Dense: 완전 연결 레이어입니다. 밀집 레이어의 각 뉴런은 이전 레이어의 모든 뉴런에 연결됩니다. 이들은 CNN 레이어가 추출한 특징을 기반으로 최종 분류를 수행하는 역할을 합니다. activation='relu'는 중간 레이어에 사용됩니다.
# • layers.Dense(5, activation='softmax'): 출력 레이어입니다.
# ㆍ 5개 유닛은 분류하려는 5가지 꽃 클래스(예: 데이지, 민들레, 장미, 해바라기, 튤립)에 해당합니다.
# ㆍ activation='softmax'는 다중 클래스 분류 문제에 사용됩니다. 
#   네트워크의 원시 출력을 확률 분포로 변환하며, 모든 클래스의 확률 합은 1이 됩니다. 
#   가장 높은 확률을 가진 클래스가 모델의 예측이 됩니다.

# 5. 모델 컴파일
# 훈련 전에 모델을 컴파일해야 합니다. 여기에는 옵티마이저, 손실 함수 및 메트릭 지정이 포함됩니다.
# • optimizer='adam': 옵티마이저는 손실을 최소화하기 위해 모델의 가중치를 조정하는 알고리즘입니다. Adam은 매우 인기 있고 효과적인 최적화 알고리즘입니다.
# • loss='sparse_categorical_crossentropy': 손실 함수는 모델의 성능을 측정합니다. 훈련 중에는 이 값을 최소화하는 것이 목표입니다.
# ㆍ sparse_categorical_crossentropy는 레이블이 정수(예: 5가지 꽃 유형에 대해 0, 1, 2, 3, 4)인 다중 클래스 분류 문제에 일반적으로 사용됩니다. 이 코드의 주석 #라벨-원핫인코딩안하려고에서 이 점을 명확히 하고 있습니다.
# • metrics=['accuracy']: 메트릭은 훈련 및 테스트 단계를 모니터링하는 데 사용됩니다. accuracy는 단순히 올바르게 분류된 이미지의 비율을 측정합니다.

# 6. 데이터셋 로드
# 이 부분은 keras.utils.image_dataset_from_directory를 사용하여 파일 시스템에서 직접 이미지 데이터를 로드합니다. 이 함수는 디렉토리 구조(예: flowers/daisy/ 이미지는 'daisy'로 레이블링됨)에서 클래스 레이블을 자동으로 추론하므로 매우 편리합니다.
# • train_dir: 이는 각 클래스에 대한 하위 디렉토리(예: flowers/daisy, flowers/dandelion 등)를 포함할 것으로 예상되는 기본 "flowers" 디렉토리의 경로입니다.
# • validation_split=0.2: Keras에게 train_dir에서 발견된 이미지를 두 부분으로 나누도록 지시합니다: 훈련용 80%, 검증용 20%.
# • subset='training' / subset='validation': 이 인수는 train_ds가 80% 훈련 데이터를 가져오고 val_ds가 20% 검증 데이터를 가져오도록 합니다.
# • seed=1234: 고정된 난수 시드를 사용하면 코드를 여러 번 실행하더라도 동일한 이미지가 일관되게 훈련 세트와 검증 세트에 할당됩니다. 이는 재현성을 위해 중요합니다.
# • image_size: 모든 로드된 이미지는 (180, 180) 픽셀로 크기가 조정됩니다. 이는 CNN의 입력 차원을 표준화하는 일반적인 전처리 단계입니다.
# • batch_size: 데이터는 배치(여기서는 한 번에 32개 이미지)로 로드되어 훈련 중에 모델에 효율적으로 공급됩니다.

# 7. 모델 훈련
# model.fit() 메서드는 실제 훈련 프로세스가 발생하는 곳입니다.
# • train_ds: 모델을 훈련하는 데 사용되는 데이터셋입니다.
# • epochs=10: 에포크는 전체 훈련 데이터셋을 한 번 완전히 통과하는 것을 의미합니다. 모델은 train_ds의 모든 배치를 10번 반복합니다.
# • validation_data=val_ds: 각 에포크 동안, 훈련 데이터를 처리한 후 모델은 이 val_ds(검증 세트)에서 성능을 평가합니다. 이는 과적합을 모니터링하는 데 중요합니다. 
#   훈련 정확도는 계속 증가하지만 검증 정확도가 떨어지기 시작한다면, 이는 과적합의 징후입니다.
# model.fit() 메서드는 history 객체를 반환하며, 이 객체에는 각 에포크의 훈련 및 검증 세트에 대한 손실 값과 메트릭 값(예: 정확도) 기록이 포함됩니다.

# 8. 모델 및 기록 저장
# • model.save("flowers_model.keras"): 이 코드는 훈련된 전체 모델(아키텍처, 가중치, 옵티마이저 상태)을 .keras 확장자 (TensorFlow 2.x에서 Keras 모델의 권장 형식)를 가진 단일 파일로 저장합니다. 이를 통해 나중에 모델을 다시 훈련하지 않고도 로드하여 사용할 수 있습니다.
# • with open("flowers_hist.pkl", "wb") as f: pickle.dump(history.history, f): 이 코드는 Python의 pickle 모듈을 사용하여 history.history 딕셔너리(각 에포크별 손실 및 정확도 값 목록 포함)를 저장합니다.
# ㆍ pickle은 Python 객체를 직렬화하고 역직렬화하는 데 사용됩니다. Python 객체(딕셔너리 등)를 바이트 스트림으로 변환(파일에 저장하기 위해)하고 나중에 바이트 스트림을 원래 객체로 다시 변환할 수 있습니다.
# ㆍ with open(...) as f: 구문은 파일을 안전하게 처리하는 파이썬의 권장 방식입니다. 오류가 발생하더라도 파일이 자동으로 닫히도록 보장합니다.
# ㆍ 파일 확장자 .hist는 사용자 정의이며, .pkl이 피클링된 파일에 더 일반적으로 사용됩니다.

# 실행 방법
# 1.TensorFlow 및 Keras가 설치되어 있는지 확인합니다.
# 2.data1/flowers 디렉토리 구조를 만듭니다. flowers 안에 각 꽃 클래스에 대한 별도의 하위 디렉토리(예: data1/flowers/daisy, data1/flowers/dandelion 등)가 있어야 합니다.
# 3.이미지 파일을 해당 클래스 하위 디렉토리에 배치합니다. 고유한 꽃 클래스의 수는 Dense 레이어의 출력 유닛 수를 결정합니다.
# 4.Python 스크립트를 실행합니다.
# 이 스크립트는 이미지 분류기 훈련을 위한 견고한 기반을 제공하며, 특히 이미지가 클래스별 폴더로 구성된 데이터셋에 유용합니다.