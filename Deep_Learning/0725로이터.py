# #250725 PM3시 로이터기사, 문자열을 받아 다중분류
# 예측하려는 것: 뉴스 기사의 주제 또는 카테고리
# 로이터 데이터셋은 여러개의 뉴스카테고리(주제)로 분류된 기사들로 구성되어 있습니다. 
#  reuters.load_data() 함수를 통해 데이터를 로드했을 때, train_labels와 test_labels에 담긴 값들은
# 특정 기사가 어떤 주제에 속하는지를 나타내는 숫자 형태의 카테고리 ID입니다.
# 예를 들어, 라벨3은 '주식시장', 4는 '곡물무역', 10은 '원유'와 같은 주제. 
# 로이터 데이터셋에는 총 46개의 다른 뉴스 주제가 있습니다.
# 입력된 뉴스 기사가 46개의 정의된 카테고리 중 어느 하나에 속하는지를 분류하는 것입니다. 
# 이는 다중 클래스 분류(Multi-class Classification) 문제에 해당합니다. 
import keras
from keras.datasets import reuters
from keras import models
from keras import layers
import tensorflow as tf
import os

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#데이터 개수 확인
print(train_data. shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
#데이터 자체도 궁금
print(train_data[:3]) #문장을 list타입으로 가져온다
print(train_labels[:3])

# 2110848/2110848 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
# (8982,)
# (8982,)
# (2246,)
# (2246,)
# [list([1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12])
#  list([1, 3267, 699, 3434, 2295, 56, 2, 7511, 9, 56, 3906, 1073, 81, 5, 1198, 57, 366, 737, 132, 20, 4093, 7, 2, 49, 2295, 2, 1037, 3267, 699, 3434, 8, 7, 10, 241, 16, 855, 129, 231, 783, 5, 4, 587, 2295, 2, 2, 775, 7, 48, 34, 191, 44, 35, 1795, 505, 17, 12])
#  list([1, 53, 12, 284, 15, 14, 272, 26, 53, 959, 32, 818, 15, 14, 272, 26, 39, 684, 70, 11, 14, 12, 3886, 18, 180, 183, 187, 70, 11, 14, 102, 32, 11, 29, 53, 44, 704, 15, 14, 19, 758, 15, 53, 959, 47, 1013, 15, 14, 19, 132, 15, 39, 965, 32, 11, 14, 147, 72, 11, 180, 183, 187, 44, 11, 14, 102, 19, 11, 123, 186, 90, 67, 960, 4, 78, 13, 68, 467, 511, 110, 59, 89, 90, 67, 1390, 55, 2678, 92, 617, 80, 1274, 46, 905, 220, 13, 4, 346, 48, 235, 629, 5, 211, 5, 1118, 7, 2, 81, 5, 187, 11, 15, 9, 1709, 201, 5, 47, 3615, 18, 478, 4514, 5, 1118, 7, 232, 2, 71, 5, 160, 63, 11, 9, 2, 81, 5, 102, 59, 11, 17, 12])]
# [3 4 3]

#get_sord_index
word_index = reuters.get_word_index()
print(type(word_index))
print(word_index.keys()) #확인후 주석처리

#word_index 내부구조확인
# 변환된 reverse_index 리스트의 처음 10개 항목을 출력하여 (인덱스, 단어) 형태를 확인
# [(인덱스, 단어), ...] 리스트 → dict. 특정인덱스에 해당단어를 reverse_index[인덱스]와 같이 직접 빠르게조회(딕셔너리가 검색속도에 유리)ㅉ
def showDictionary(cnt):
    i = 0
    for key in word_index:
        if i >= cnt:
            break
        print(key, word_index[key])
        i += 1

showDictionary(10) #영단어:인덱스
reverse_index = [(value, key) for (key, value) in word_index.items()]
for i in range(0, 10):
    print(reverse_index[i])
reverse_index = dict(reverse_index)
#dict으로 바꿔야 한번에 검색함

#id에 해당하는 train_data 시퀀스를 가져와 실제 문장으로 변환
#케라스만든 사람들이 0~3번까지는 특수목적으로 인덱스4부터 쓸모있음
def changeSentence(id):
    sequence = train_data[id]
    sentence = ' '.join(reverse_index.get(i-3, '*') for i in sequence) #⭐핵심변환로직
    #              합치자         워드인덱스에 3을 더해서저장  *:1만개만 가져오니까 없는단어가 있을경우 2번째인자로 출력. 잘보이라고 *표기
    print(sentence)

#원하는대로 함수생성

print("--- Processing sentences 모든 훈련 데이터를 문장으로 변환하여 출력 ---")
#for문으로 다회호출가능
# for i in range(len(train_data)):
# 모든 문장을 다 출력하면 너무 많을 수 있으므로, 일부만 테스트로 출력하는 것을 권장합니다.
for i in range(10): # 처음 10개의 훈련 기사만 출력하여 확인
    changeSentence(i)

#train_data : 시퀀스의 배열
#원핫인코딩 - 내부함수 말고 한번 만들어보자. 1만개까지만 불러오자
#단어인덱스리스트로 되어 있는 시퀀스를 신경망이 처리할수있는 원-핫 인코딩된 벡터로 변환.
import numpy as np
def vectorize_sentences(sequences, dimensions=10000):
    # 문장 개수 * 10000개의 2D 배열을 0으로 채워 생성합니다.
    results = np.zeros((len(sequences), dimensions)) #zeros가 매개변수고 tuple받아감
    for i, sequence_data in enumerate(sequences): 
        # 각 시퀀스에 대해 해당 인덱스를 1로 설정합니다.
        # Numpy의 고급 인덱싱을 사용하여 해당 인덱스를 1로 설정 (효율적)
        # 예를 들어, sequence_data가 [1, 4, 11]이면 results[i, 1], results[i, 4], results[i, 11]이 1이 됩니다.
        results[i, sequence_data] = 1 
    return results

#시퀀스 → 원핫인코딩
X_train = vectorize_sentences(train_data)
X_test = vectorize_sentences(test_data)
print(X_train[:3])
print("X_train (원-핫 인코딩) shape:", X_train.shape) # 형태 확인
print("X_test (원-핫 인코딩) shape:", X_test.shape) # 형태 확인

#훈련셋과 검증셋 분할. split함수써도됨
print(len(train_data)) # 실제 train_data의 길이 출력: 8982
# 참고: 로이터 훈련 데이터는 8982개, 테스트 데이터는 2246개입니다.
# 10000개로 분할하는 것은 train_data의 크기를 넘어섭니다.
X_val = X_train[:1000] #검증셋1000개
X_train = X_train[1000:] #훈련셋은 8982-1000개
y_val = train_labels[:1000]
y_train = train_labels[1000:]

#모델 생성
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # input_shape 명시
model.add(layers.Dropout(0.5)) # 과적합방지추가: 50%의 뉴런을 드롭아웃
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5)) # 과적합방지추가: 50%의 뉴런을 드롭아웃
# model.add(layers.Dense(16, activation='relu')) #과적합방지추가:복잡도줄이기:Dense레이어개수 줄이거나 각레이어의 뉴런(노드)수를 줄여서 단순화
model.add(layers.Dense(46, activation='softmax')) # 46개의 주제를 예측
# Reuters는 46개 클래스 다중 분류 문제이므로 출력 레이어 변경노드 수: 46 (클래스 개수)

print("--- 모델 요약 출력 (모델 구조 확인용) ---")
# # model.summary() #확인하고 주석처리
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 16)                  │         160,016 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 16)                  │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 16)                  │             272 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout_1 (Dropout)                  │ (None, 16)                  │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 46)                  │             782 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 161,070 (629.18 KB)
#  Trainable params: 161,070 (629.18 KB)
#  Non-trainable params: 0 (0.00 B)


# 과적합방지: Early Stopping: Keras의 콜백사용해 검증손실이 더이상 개선않될때 자동학습중단
import keras.callbacks
from keras import models, layers
# Early Stopping 콜백 정의 (함수 밖으로 빼내고 변수에 바로 할당) ---
# ModelCheckpoint 콜백 정의
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath="로이터_사전학습.keras", 
        save_best_only=True,
        monitor='val_loss', # 검증 손실을 기준으로 최적 모델 저장
        verbose=1 # 학습 중 저장 시 메시지 출력
    ),
    # EarlyStopping 콜백 추가 (선택 사항이지만 과적합 방지에 매우 효과적)
    keras.callbacks.EarlyStopping(
        monitor='val_loss', # 검증 손실을 모니터링
        patience=5, # 5 에포크 동안 개선이 없으면 중단
        restore_best_weights=True # 학습 중 가장 좋았던 가중치를 복원
    )
]
# 컴파일 :모델학습fit/평가evaluate전 컴파일 필수, 안하니 에러
# 모델이 "어떻게 학습할 것인가"를 정의하는 단계
# model.compile() 함수는 Keras 모델이 훈련될 준비를 하도록 설정하는 중요한 단계이기 때문입니다. 이 함수는 다음 세 가지 핵심 요소를 모델에게 알려줍니다:
# • 옵티마이저 (Optimizer):
# 모델의 가중치(weights)를 어떻게 업데이트할 것인지를 결정하는 알고리즘입니다. 학습 과정에서 모델의 예측 오차를 줄이기 위해 가중치를 조절하는 "학습 전략"이라고 생각할 수 있습니다.
# 예시: rmsprop, adam, sgd 등
# model.compile()을 하지 않으면 Keras는 어떤 옵티마이저를 사용해야 할지 모르기 때문에 가중치를 업데이트할 수 없습니다.
# • 손실 함수 (Loss Function):
# 모델의 예측이 실제 정답과 얼마나 차이가 나는지를 측정하는 함수입니다. 모델이 훈련 중에 최소화하려고 노력하는 "오차"를 정의합니다.
# 예시: sparse_categorical_crossentropy, binary_crossentropy, mse (평균 제곱 오차) 등
# model.compile()을 하지 않으면 Keras는 예측 오차를 어떻게 계산해야 할지 모르므로, 이 오차를 기반으로 가중치를 업데이트할 수 없습니다.
# • 평가 지표 (Metrics):
# 훈련 및 테스트 과정에서 모델의 성능을 측정하고 모니터링하기 위한 지표입니다. 손실 함수와는 다르게, 모델 훈련에 직접적으로 사용되지는 않지만, 사람이 모델의 성능을 이해하는 데 도움을 줍니다.
# 예시: accuracy (정확도), precision (정밀도), recall (재현율) 등
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#모델 학습 및 히스토리 저장
print("\n--- 모델 학습 시작 ---")
history = model.fit(X_train, y_train, epochs=20,  #초기엔 과적합방지해야하니 에포크줄여서
                    batch_size=512, validation_data=(X_val, y_val),
                    callbacks=callbacks_list) #과적합방지위해 콜백함수
print("--- 모델 학습 완료 ---")
# Epoch 1/20
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 3s 35ms/step - accuracy: 0.1153 - loss: 3.6105 - val_accuracy: 0.5040 - val_loss: 3.0545
# Epoch ...
# Epoch 19/20
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.8355 - loss: 0.6652 - val_accuracy: 0.7340 - val_loss: 1.2044
# Epoch 20/20
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.8359 - loss: 0.6395 - val_accuracy: 0.7340 - val_loss: 1.2037

#평가하기
print("\n--- 모델 평가 시작 ---")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=0)
print(f"훈련셋  손실: {train_loss:.4f}, 훈련셋 정확도: {train_acc:.4f}")
print(f"테스트셋 손실: {test_loss:.4f}, 테스트셋 정확도: {test_acc:.4f}")
#1차 모델평가
# • 훈련셋  손실: 0.5975, 훈련셋 정확도: 0.8430
# • 테스트셋 손실: 1.2734, 훈련셋 정확도: 0.7159
# 훈련셋손실 낮지만, 테스트셋손실이 훨씬높다. → 과적합(Overfitting)
# 이는 모델이 훈련데이터를 너무 '외워서' 새데이터(테스트셋)는 성능떨어집니다.
# 마치 시험범위만 달달 외워서 아는문제만 잘풀고, 응용문제나 처음보는문제는 못푸는것.

# 과적합방지 + 테스트셋손실 낮추기위한 일반적인 방법: 모델의 일반화 성능을 높이는 것
# • 에포크(Epoch) 줄이기: 현재 50 에포크로 학습하고 계신데, 이는 다소 많을 수 있습니다. 모델이 훈련 데이터를 지나치게 학습하기 전에 학습을 멈춰야 합니다. 10~20 에포크 사이에서 검증 손실(validation loss)이 다시 높아지는 시점을 찾아 그쯤에서 훈련을 멈추는 것이 좋습니다.
# • 드롭아웃(Dropout) 추가: 신경망 레이어 사이에 layers.Dropout()을 추가하여 학습 시 특정 뉴런을 무작위로 비활성화합니다. 
# 이는 모델이 특정 뉴런에 과도하게 의존하는 것을 방지하고 일반화 능력을 향상시킵니다..

#과적합방지 추가후 2차 모델평가
# Epoch 1/20
# 12/16 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.1010 - loss: 3.7412
# Epoch 1: val_loss improved from inf to 3.29337, saving model to 로이터_사전학습.keras
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 36ms/step - accuracy: 0.1176 - loss: 3.7055 - val_accuracy: 0.4300 - val_loss: 3.2934
# ...
# Epoch 20/20
# 14/16 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5492 - loss: 1.6215 
# Epoch 20: val_loss improved from 1.45278 to 1.43692, saving model to 로이터_사전학습.keras
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.5496 - loss: 1.6213 - val_accuracy: 0.6340 - val_loss: 1.4369
# 훈련셋  손실: 1.2569, 훈련셋 정확도: 0.6673
# 테스트셋 손실: 1.4787, 테스트셋 정확도: 0.6273
# 모델이 막 학습을 시작했습니다. 첫 에포크라 아직 훈련 데이터와 검증 데이터 모두에서 정확도가 낮고 손실은 높게 나타나는 것이 일반적입니다.
# 특히 훈련 정확도(0.1176)보다 검증 정확도(0.4300)가 훨씬 높은 것은 긍정적인 신호일 수 있습니다. 이는 훈련셋이 무작위로 초기화된 모델에 비해 상대적으로 더 어렵거나, 아니면 훈련 과정 초기에 데이터 셋 분배에 따른 우연의 결과일 수 있습니다. 앞으로 에포크가 진행되면서 훈련 정확도와 검증 정확도 모두 점진적으로 상승해야 합니다.
# ModelCheckpoint 콜백이 잘 작동하여 첫 번째 모델이 저장되었습니다.
# 이제 다음 에포크가 진행되면서 accuracy와 val_accuracy는 높아지고 loss와 val_loss는 낮아지는지 계속 지켜보시면 됩니다. 만약 val_loss가 더 이상 낮아지지 않고 오히려 높아지기 시작한다면, 과적합이 발생하고 있다는 신호이므로 EarlyStopping 콜백이 학습을 중단시킬 것입니다.


#예측
pred = model.predict(X_test)
print("첫 10개 예측 확률 배열:\n", pred[:10]) #라벨이 1이 될 확률 준다

# 다중 클래스 분류에서는 특정 임계값(0.5)을 기준으로 0 또는 1로 바꾸는 것이 아니라, 46개 확률 중 가장 높은 값을 가진 클래스의 인덱스를 선택해야 합니다.
# 다중 클래스 분류를 위한 예측 결과 변환
# 각 샘플에 대해 가장 높은 확률을 가진 클래스의 인덱스를 선택
predicted_classes = np.argmax(pred, axis=1) # axis=1은 각 행(샘플)에 대해 최대값의 인덱스를 찾음

print("\n--- 예측 결과와 실제 라벨 비교 ---")
for i in range(0, 40):
    print(f"예측: {predicted_classes[i]}, 실제: {test_labels[i]}")


#훈련과 검증 정확도 시각화
import matplotlib.pyplot as plt
history_dict = history.history
acc = history_dict['accuracy'] #훈련셋 정확도
val_acc = history_dict['val_accuracy'] #검증셋 정확도
loss = history_dict['loss']  #훈련셋 손실
val_loss = history_dict['val_loss'] #검증셋 손실

#정확도 그래프
length = range(1, len(acc)+1 ) #X축만들기
plt.figure(figsize=(10,5))
plt.plot(length, acc, 'bo', label='Training acc')
plt.plot(length, val_acc, 'b', label='Validation acc')
plt.title("Reuters Training and Validation acc")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.grid(True) #격자
plt.show()

#손실그래프 추가
plt.figure(figsize=(10, 5))
plt.plot(length, loss, 'ro-', label='Training loss') #빨간원마커와 실선
plt.plot(length, val_loss, 'r-', label='Validation loss')
plt.title("로이터 Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- 모델 학습 시작 ---
# Epoch 1/20
# 15/16 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.0789 - loss: 3.7686
# Epoch 1: val_loss improved from inf to 3.44472, saving model to 로이터_사전학습.keras
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 2s 41ms/step - accuracy: 0.0845 - loss: 3.7591 - val_accuracy: 0.5300 - val_loss: 3.4447
# Epoch 2/20
# 13/16 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.2224 - loss: 3.4533
# Epoch 2: val_loss improved from 3.44472 to 3.00910, saving model to 로이터_사전학습.keras
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step - accuracy: 0.2317 - loss: 3.4320 - val_accuracy: 0.5320 - val_loss: 3.0091
# ...
# Epoch 20/20
# 14/16 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.5538 - loss: 1.6291 
# Epoch 20: val_loss improved from 1.39667 to 1.38313, saving model to 로이터_사전학습.keras
# 16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5547 - loss: 1.6307 - val_accuracy: 0.6640 - val_loss: 1.3831
# --- 모델 학습 완료 ---

# --- 모델 평가 시작 ---
# 훈련셋  손실: 1.2246, 훈련셋 정확도: 0.6979
# 테스트셋 손실: 1.4260, 테스트셋 정확도: 0.6567
# 71/71 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
# 첫 10개 예측 확률 배열:
#  [[3.12611119e-05 1.46589661e-03 3.16088299e-05 9.38057601e-01
#   3.11965477e-02 3.11499116e-06 3.31055562e-05 2.02851934e-05
#   1.26582582e-03 2.07187768e-04 1.32385496e-04 5.15275449e-03
#   1.09335539e-04 1.00618391e-03 1.54545924e-05 3.57258500e-06
#   1.11670885e-02 7.97357643e-05 2.25091280e-05 5.52696595e-03
#   3.44445836e-03 2.25144002e-04 5.15933607e-06 6.52693634e-05
#   5.20566355e-05 9.95396476e-05 7.32217495e-06 1.27550802e-05
#   2.40152403e-05 1.84433720e-05 5.34888313e-05 1.69593750e-05
#   1.74985580e-05 1.62156211e-05 3.83654951e-05 2.08393385e-06
#   1.13440794e-04 3.41858868e-06 2.03372038e-05 1.07625010e-05
#   8.25415045e-05 1.05821367e-04 8.37143398e-06 1.93161250e-05
#   6.83400970e-07 8.13772476e-06]
# ...
#  [4.54763242e-04 9.15670954e-03 3.71732516e-04 8.27525139e-01
#   4.68823537e-02 8.19780107e-05 3.22624022e-04 3.71174392e-04
#   6.90935925e-03 1.37100485e-03 1.31438102e-03 2.37898882e-02
#   1.16980437e-03 5.98122599e-03 2.48673488e-04 8.60359360e-05
#   2.90522687e-02 6.75319578e-04 3.04075045e-04 2.01323032e-02
#   1.30255911e-02 1.48992520e-03 1.19002332e-04 7.21977733e-04
#   6.91559864e-04 1.09061005e-03 1.29277614e-04 1.77153051e-04
#   3.53401905e-04 3.69827816e-04 5.67379291e-04 2.47745571e-04
#   2.42729569e-04 2.29973943e-04 5.12866362e-04 5.22809169e-05
#   1.08815089e-03 6.30764189e-05 3.19781509e-04 1.69815554e-04
#   6.99277094e-04 7.81357230e-04 1.92495077e-04 2.95025471e-04
#   2.31049944e-05 1.45735481e-04]]

# --- 예측 결과와 실제 라벨 비교 ---
# 예측: 3, 실제: 3
# 예측: 1, 실제: 10
# 예측: 1, 실제: 1
# 예측: 4, 실제: 4
# 예측: 16, 실제: 4
# 예측: 3, 실제: 3
# 예측: 3, 실제: 3
# 예측: 3, 실제: 3
# 예측: 3, 실제: 3
# 예측: 3, 실제: 3
# 예측: 1, 실제: 5
# 예측: 4, 실제: 4
# 예측: 1, 실제: 1
# 예측: 3, 실제: 3
# 예측: 1, 실제: 1
# 예측: 3, 실제: 11
# 예측: 4, 실제: 23
# 예측: 3, 실제: 3
# 예측: 19, 실제: 19
# 예측: 3, 실제: 3
# 예측: 3, 실제: 8
# 예측: 3, 실제: 3
# 예측: 3, 실제: 3
# 예측: 4, 실제: 3
# 예측: 4, 실제: 9
# 예측: 3, 실제: 3
# 예측: 4, 실제: 4
# 예측: 4, 실제: 6
# 예측: 1, 실제: 10
# 예측: 3, 실제: 3
# 예측: 3, 실제: 3
# 예측: 1, 실제: 10
# 예측: 4, 실제: 20
# 예측: 3, 실제: 1
# 예측: 19, 실제: 19
# 예측: 4, 실제: 4
# 예측: 3, 실제: 40
# 예측: 1, 실제: 1
# 예측: 4, 실제: 4
# 예측: 3, 실제: 3