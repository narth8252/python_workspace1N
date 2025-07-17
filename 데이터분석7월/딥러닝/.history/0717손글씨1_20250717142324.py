# Ctrl + Shift + P → "Python: Select Interpreter" 입력해서 선택
# C:\ProgramData\Anaconda3\envs\deeplearning\python.exe
#conda activate deeplearning

# 손글씨 숫자(MNIST)를 분류하는 딥러닝 모델(우편번호)
# 숫자 이미지를 넣으면, 0~9 중에서 어떤 숫자인지 예측할 수 있어.
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(1234)

#1.MNIST 데이터셋 로드 (mnist.load_data())
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(type(train_images), type(train_labels)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
# train_images, train_labels 둘 다 numpy.ndarray 타입으로 잘 불러왔다는 뜻이야.

# 2. 데이터 전처리 (reshape + 정규화,스케일링)
# 3D → 2D 배열로 바꾸고, 픽셀 값 0255 → 01로 바꿨어.
train_images = train_images.reshape(train_images.shape[0], 28 * 28)
train_images = train_images.astype(float) / 255
test_images = test_images.reshape(test_images.shape[0], 28 * 28)
test_images = test_images.astype(float) / 255

print(train_images.shape, train_images.min(), train_images.max())
# 훈련데이터shape가 (60000, 784) 0.0 1.0 (최소값0.0, 최대값1.0)으로 정상적인전처리
# 훈련데이터가 60000개, 각 이미지가 784(28*28) 픽셀로 1차원 배열화됐고, 값도 0~1 범위로 잘 정규화됐어.
print(test_images.shape, test_images.min(), test_images.max())
# 테스트데이터가 (10000, 784)형태이고 0.0 1.0 로 훈련데이터와 동일하게 전처리됐어.

# 3. 모델생성 후 실행
# 입력층 64개 뉴런, 출력층 10개 뉴런(숫자 0~9 클래스)
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(28*28,)),
    layers.Dense(10, activation='softmax')
])
# 모델 구조를 출력해보자
model.summary() #모델구조출력
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param # (파라미터수)
# =================================================================
#  dense (Dense)  1st입력층       (None, 64)편향            50240   #첫번째Dense층(입력층)파라미터수: 50,240개(입력 784*64+편향64)
#  dense_1 (Dense)2nd출력층       (None, 10)편향             650    #두번째Dense층(출력층)파라미터수: 650개(64*10+편향10)
# =================================================================
# Total params: 50,890      #총파라미터: 50,890개 
# Trainable params: 50,890  #모두 학습가능한 파라미터임
# Non-trainable params: 0
# _________________________________________________________________

# 4. 모델 컴파일
# → 숫자 라벨 그대로 쓸 거니까 손실함수는 sparse_categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 모델 학습 시작
# → 64개씩 5번 학습하고, 매 에폭마다 섞어서 학습하는 중
history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size=64,
    epochs=5,
    verbose=1,
    shuffle=True
)
# ✅ Epoch (에폭): 전체 학습 데이터를 한 번 다 학습한 횟수야. 너는 5번 돌렸어.
# ✅ Loss (로스): 모델이 틀린 정도야. 작을수록 좋음.
# ✅ Accuracy (정확도): 모델이 맞춘 비율이야. 클수록 좋음.
# ✅ 모델을 훈련할 때, 전체 데이터를 훈련용(train) 과 검증용(validation) 으로 나눠.
# 훈련용 데이터는 모델이 직접 보고 학습하는 데이터야.
# 검증용 데이터(val) 는 모델이 직접 보지 않은 데이터로, 모델이 새로운 데이터를 얼마나 잘 예측할 수 있는지를 확인하는 데 써.
# val_로 시작하는 건 검증 데이터에 대한 성능이야 (모델이 실제로 얼마나 잘 일반화되는지 확인하는 용도).
# • val_loss: 검증 데이터에 대한 손실값
# • val_accuracy: 검증 데이터에 대한 정확도
# 훈련 정확도만 보면 과적합(overfitting)이 일어났는지 알 수 없어.
# 검증 정확도가 높아야 진짜 "실전에서도 잘 예측하는 모델"이라고 할 수 있어. 
# ✅ loss = 손실(오차의 정도)
# 모델이 예측한 값과 실제 정답 사이의 차이(오차) 를 수치로 표현한 거야.
# 예를 들어 숫자 7을 이미지로 넣었는데, 모델이 5라고 예측하면 차이가 있으니까 loss가 커져.
# 모델은 loss를 줄이기 위해 학습해.
# 📉 낮을수록 좋다! 모델이 정답에 가까워지고 있다는 의미야.
# 하지만 loss가 낮다고 정확도가 반드시 높은 건 아냐, 특히 다중 클래스 분류일 때는 정확도와 함께 같이 봐야 해.   

# ________학습과정 출력_________________________________________________________
# Epoch 1/5                                                               훈련정확도 90.01%                          검증정확도 93.96%
# 938/938 [==============================] - 5s 4ms/step - loss: 0.3579 - accuracy: 0.9001 - val_loss: 0.2031 - val_accuracy: 0.9396
# Epoch 2/5
# 938/938 [==============================] - 3s 3ms/step - loss: 0.1819 - accuracy: 0.9485 - val_loss: 0.1528 - val_accuracy: 0.9532
# Epoch 3/5
# 938/938 [==============================] - 3s 3ms/step - loss: 0.1385 - accuracy: 0.9596 - val_loss: 0.1285 - val_accuracy: 0.9637
# Epoch 4/5
# 938/938 [==============================] - 3s 3ms/step - loss: 0.1119 - accuracy: 0.9672 - val_loss: 0.1125 - val_accuracy: 0.9657
# Epoch 5/5
# 938/938 [==============================] - 3s 3ms/step - loss: 0.0939 - accuracy: 0.9719 - val_loss: 0.1053 - val_accuracy: 0.9679
# • 과적합 없음: train과 val 정확도 차이 거의 없음.
# • 점진적인 성능 향상: 정확히 잘 학습되고 있음.
# • 5 epoch만에 96.8% → 굉장히 우수한 성능 (MNIST 기준).
# 모델이 훈련 데이터를 점점 더 잘 맞추고 있고,
# 검증 데이터에서도 높은 성능을 보여주고 있어서,
# 과적합 없이 아주 잘 훈련되고 있다는 뜻이야.

# ✅지금까지 모델을 잘 학습했으니까, 이제 그다음 단계는 "이 모델을 실전에 써먹는 준비"야.
#     단계	            설명	                                             추천여부
# 1️⃣ 예측(Predict)	    숫자 이미지 넣고, 모델이 어떤 숫자인지 예측하게 하기	   ✅ 꼭 해봐야 함
# 2️⃣ 시각화(Visualize)	예측 결과를 이미지로 보여줘서 “진짜 맞췄나?” 눈으로 확인	👍 재밌고 직관적
# 3️⃣ 모델 저장(Save)	지금 훈련한 모델을 .h5나 .keras 파일로 저장	              나중에 재사용 가능
# 4️⃣ 정확도향상(Tuning)정확도 97%도 좋지만, 더 높이고 싶다면 하이퍼파라미터 조정    도전해보고 싶다면

# model = keras.Sequential([
# #2-1.입력층 설계
# #layers.Dense(출력값의개수, 활성화함수, 입력데이터의 크기-생략가능)
# #출력값개수? 저 계층을 나왔을때 가져올 가중치들의 개수.
# #           내마음대로 너무 크게 주면, 메모리부족과 과대적합 문제,
# #           적당히 2의 배수로들 준다
#     layers.Dense(64, activation='relu' ),
# #2-2 중간에 다른층 추가 가능
# #2-3 출력층,마지막층은 라벨에 맞추기.결과얻기위한층
# #    손으로쓴 숫자이니 0~9 중에 하나여야한다.
# #   딥러닝분류는 출력데이터를 확률로 반환
# #   예)[0.1,0.1,0.05,0.7...] 결과는3으로 판단
# #   각층 거치며 나오는값들은 실제확률이 아닌 엄청큰값들.
# #   이를 모두합해 1이되는 확률로 전환해야하는데 이 함수가 softmax 함수
# #   다중분류의 출력층의 활성화함수는 무조건 softmax함수
# #내용 외워질때까지 연습해야함
#     layers.Dense(10, activation='softmax') #출력값개수, 활성화함수
# #회귀랑 이진분류,다중분류 다 다르게 작성.
# #회귀는 출력결과1개만
# #회귀의경우 출력층: layers.Dense(1, activation='linear')
# #이진분류의경우 출력층: layers.Dense(1, activation='sigmoid')
# ])

# #compile() 메서드로 모델을 컴파일한다.(모델학습시킬준비한다)
# # 모델을 컴파일할 때는 다음같은 매개변수 지정
# #loss: 손실함수, 모델이 예측한 값과 실제 값의 차이를 계산하는 함수
# #       다중분류는 sparse_categorical_crossentropy, 이진분류는 binary_crossentropy
# #       회귀는 mean_squared_error
# #optimizer: 최적화 알고리즘, adam, sgd 등
# #metrics: 모델의 성능을 평가할 때 사용할 평가 지표, accuracy(정확도) 등
# #       이진분류는 accuracy, 다중분류는 sparse_categorical_accuracy

# #3.데이터
# # 머신러닝 vs 딥러닝 라벨 인코딩 정리 메모
# # 머신러닝 모델 (랜덤포레스트, SVM 등)
# # → 숫자형 라벨 사용 (라벨 인코딩)
# # → y = [0, 1, 2, ...] 형식 그대로 사용 가능
# # 쌤:머신러닝은 라벨을 원핫인코딩, 딥러닝은 원핫인코딩자동.
# # 딥러닝 모델 (Keras, TensorFlow 등)
# # loss = 'categorical_crossentropy' → 원핫인코딩 필요: y = [[1,0,0], [0,1,0], ...]
# # loss = 'sparse_categorical_crossentropy' → 숫자형 라벨 그대로 사용 가능
# # 딥러닝이 라벨을 내부적으로 자동 원핫 인코딩하지는 않음 → 인코딩은 사용자가 직접 처리하거나 손실 함수로 제어
