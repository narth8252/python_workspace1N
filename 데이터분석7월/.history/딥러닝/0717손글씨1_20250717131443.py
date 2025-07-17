# Ctrl + Shift + P → "Python: Select Interpreter" 입력해서 선택
# C:\ProgramData\Anaconda3\envs\deeplearning\python.exe
#conda activate deeplearning

from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(1234)

#1.데이터가져오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(type(train_images), type(train_labels))
#첫시작시 7만개 다운로드


#2.딥러닝 모델을 만든다.
from tensorflow.keras import models, layers

#3.네트워크 또는 모델을 만든다.
#keras.Sequential 로 모델을 만드는데 매개변수로 list타입안에 레이어 객체를 전달한다

model = keras.Sequential([
#2-1.입력층 설계
#layers.Dense(출력값의개수, 활성화함수, 입력데이터의 크기-생략가능)
#출력값개수? 저 계층을 나왔을때 가져올 가중치들의 개수.
#           내마음대로 너무 크게 주면, 메모리부족과 과대적합 문제,
#           적당히 2의 배수로들 준다
    layers.Dense(64, activation='relu' ),
#2-2 중간에 다른층 추가 가능
#2-3 출력층,마지막층은 라벨에 맞추기.결과얻기위한층
#    손으로쓴 숫자이니 0~9 중에 하나여야한다.
#   딥러닝분류는 출력데이터를 확률로 반환
#   예)[0.1,0.1,0.05,0.7...] 결과는3으로 판단
#   각층 거치며 나오는값들은 실제확률이 아닌 엄청큰값들.
#   이를 모두합해 1이되는 확률로 전환해야하는데 이 함수가 softmax 함수
#   다중분류의 출력층의 활성화함수는 무조건ㄴ softmax함수
#내용 외워질때까지 연습해야함
    layers.Dense(10, activation='softmax') #출력값개수, 활성화함수
#회귀랑 이진분류,다중분류 다 다르게 작성.
#회귀는 출력결과1개만
#회귀의경우 출력층: layers.Dense(1, activation='linear')
#이진분류의경우 출력층: layers.Dense(1, activation='sigmoid')
])

#compile() 메서드로 모델을 컴파일한다.(모델학습시킬준비한다)
# 모델을 컴파일할 때는 다음같은 매개변수 지정
#loss: 손실함수, 모델이 예측한 값과 실제 값의 차이를 계산하는 함수
#       다중분류는 sparse_categorical_crossentropy, 이진분류는 binary_crossentropy
#       회귀는 mean_squared_error
#optimizer: 최적화 알고리즘, adam, sgd 등
#metrics: 모델의 성능을 평가할 때 사용할 평가 지표, accuracy(정확도) 등
#       이진분류는 accuracy, 다중분류는 sparse_categorical_accuracy

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#3.데이터관련작업(3D → 2D 변환, 정규화 스케일링:딥러닝은필수)
# train_images = train_images.reshape((60000, 28 * 28))
# test_images = test_images.reshape((10000, 28 * 28))
train_images = train_images.reshape(train_images.shape[0], 28 * 28)
train_images = train_images.astype(float) / 255 # 정규화: 0~255 → 0~1로 변환
test_images = test_images.reshape(test_images.shape[0], 28 * 28)
test_images = test_images.astype(float) / 255

# loss = 'sparse_categorical_crossentropy' # 손실함수안쓰고 categorical_crossentropy로 하면
# 원핫인코딩된 라벨이 필요하다. 즉, y = [[1,0,0], [0,1,0], ...] 형식으로 변환해야함
# sparse_categorical_crossentropy를 사용하면 라벨을 숫자형으로 그대로 사용할 수

#4.모델 학습: 학습과정내용 history 객체로 반환
# train_images: 훈련 데이터, train_labels: 훈련 데이터의 라벨
#fit() 메서드로 모델을 학습시킨다.
history = model.fit(train_images,   #훈련 데이터:X,독립변수,입력값
                    train_labels,   #훈련 데이터의 라벨:y, 종속변수, 출력값
                    validation_data=(test_images, test_labels), #검증용 데이터
                    batch_size=64, # 배치사이즈: 한번에 몇개씩 학습할지
                    # 데이터 메모리 불러올때 크기너무크면 메모리부족, 너무작으면 학습시간 오래걸림(속도느림)
                    # 전체 데이터를 64개씩 나눠서 학습. batch_size=64만큼 불러서 학습끝난1바퀴가 1에포크
                    # epochs=5면 64개씩 5번 학습
                    epochs=5, # 에폭: 전체 데이터셋을 몇번 반복해서 학습할지 학습회수, epochs=5면 64개씩 5번 학습
                    verbose=1, # 학습과정 출력 여부, 0: 출력안함, 1: 진행상황 출력, 2: 에폭별로 출력
                    shuffle=True, # 에폭마다 데이터 섞기 여부, True: 섞음
                    # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)], # 조기 종료 콜백
                    # callbacks=[keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True)],
                    validation_split=0.2) # 검증용 데이터 비율: 20%

# 머신러닝 vs 딥러닝 라벨 인코딩 정리 메모
# 머신러닝 모델 (랜덤포레스트, SVM 등)
# → 숫자형 라벨 사용 (라벨 인코딩)
# → y = [0, 1, 2, ...] 형식 그대로 사용 가능

# 딥러닝 모델 (Keras, TensorFlow 등)
# loss = 'categorical_crossentropy' → 원핫인코딩 필요: y = [[1,0,0], [0,1,0], ...]
# loss = 'sparse_categorical_crossentropy' → 숫자형 라벨 그대로 사용 가능
# 딥러닝이 라벨을 내부적으로 자동 원핫 인코딩하지는 않음 → 인코딩은 사용자가 직접 처리하거나 손실 함수로 제어
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)  # y = [0, 1, 2]

# 딥러닝 (원핫)
from tensorflow.keras.utils import to_categorical
model.compile(loss='categorical_crossentropy', ...)
model.fit(X, to_categorical(y))  # y = [[1,0,0], [0,1,0], …]

# 딥러닝 (숫자 그대로)
model.compile(loss='sparse_categorical_crossentropy', ...)
model.fit(X, y)  # y = [0, 1, 2]
