#250725 PM1시
#IMDb 영화 리뷰 데이터셋을 이용한 감성 분류 딥러닝 모델 학습 및 평가과정, 예측결과, 문자열을 받아 이진분류
# # IMDb 특정 영화 리뷰가 '긍정적(Positive)'인지 아니면 '부정적(Negative)'인지 감성(Sentiment)을 예측.
# 긍정/부정 분류는 주로 감성 분석(Sentiment Analysis)에서 다루는 이진 분류(Binary Classification) 문제입니다.

# 이 딥러닝 모델의 감성예측 과정
#1.입력: 모델은 전처리된 영화 리뷰를 입력으로 받습니다. 
# 이 리뷰는 원래 텍스트였지만, 모델에 들어가기 전에는 각 단어가 숫자로 변환된 시퀀스(예: [1, 14, 22, ...])로, 
# 다시 이 숫자들이 원-핫 인코딩된 10,000차원의 벡터 형태로 변환됩니다. 
# 각 벡터는 특정 단어들이 리뷰에 나타났는지 여부를 나타냅니다.

#2.모델학습: 모델은 수많은 훈련용 영화 리뷰(텍스트)와 그에 해당하는 정답 감성(긍정/부정)을 학습합니다. 
# 이 과정에서 모델은 어떤 단어 조합이나 패턴이 긍정적인 감성을 나타내고, 어떤 패턴이 부정적인 감성을 나타내는지 
# 스스로 규칙을 찾아냅니다.

#3.예측출력: 학습후 모델에 새(학습에 사용되지 않은)영화리뷰를 주면, 모델은 해당리뷰가 '긍정적일 확률'을 출력
# 이 확률은 0과 1 사이값으로 나타납니다. 0.999면 긍정리뷰확률이 99.9%라고 예측
# 0.0000005면 리뷰가 긍정확률이 거의없다는뜻이므로, 모델은 부정적이라고 판단

#4.최종분류: 이 확률값 기준으로 최종적으로 '긍정' 또는 '부정'으로 분류. 
# 일반적으로 0.5를 기준으로, 확률 ≥0.5 이면 긍정(1), 확률<0.5면 부정(0)으로 분류

#5.왜 이런 예측을 할까요?
# 영화리뷰 감성분류는 자연어 처리(Natural Language Processing, NLP)분야의 고전문제. 
# 영화 스트리밍 서비스, 상품 리뷰 사이트 등에서 사용자가 작성한 방대한 양의 텍스트 데이터를 자동으로 분석하여, 
# 해당 리뷰가 긍정적인지 부정적인지를 파악하는 데 활용될 수 있습니다. 
# 이를 통해 기업은 고객의 피드백을 빠르게 이해하거나, 추천 시스템의 정확도를 높이는 등의 이점을 얻을 수 있습니다.
# 따라서 이코드는 IMDb영화리뷰의 텍스트데이터를 분석하여 그안에 담긴 감성(긍정 또는 부정)을 기계학습(딥러닝)을 통해 자동예측시스템을 구축하고 있는 것입니다.

import keras
from keras.datasets import imdb
from keras import models
from keras import layers
import tensorflow as tf
import os

#케라스 입장에서 문자열 데이터들을 어떤 형태로 numpy배열로 만들었는지를 보고
#imdb 데이터셋 => numpy 배열로 바꿔서 온거
#문자열들을 어떤식으로 numpy 배열로 바꿀 것인가? (다음주에)
#영화평들을 다 읽어서 => numpy배열로 바꾼다.(케라스)
#빈도수로 파악할때 자주 쓰는 단어 10000 개만 가져다 쓰겠다
#num_words=10000 :빈도수를 기반으로 해서 자주 쓰는 단어 만개만 가져다 쓰겠다
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#데이터 개수 확인
print(train_data. shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
#데이터 자체도 궁금
print(train_data[:3]) #문장을 list타입으로 가져온다
print(train_labels[:3])

#데이터를 시퀀스로 바꿔야하는데 담주에 학습
#get_sord_index
word_index = imdb.get_word_index()
print(type(word_index))
# print(word_index.keys()) #확인후 주석처리

#word_index 내부구죠확인
def showDictionary(cnt):
    i = 0
    for key in word_index:
        if i >= cnt:
            break
        print(key, word_index[key])
        i += 1
# fawn 34701
# tsukino 52006
# nunnery 52007
# sonja 16816
# vani 63951
# woods 1408
# spiders 16115
# hanging 2345
# woody 2289
# trawling 52008

showDictionary(10) #영단어:인덱스
#단어들어오는 순서대로 번호붙임  I like star
#                           [0, 1,  2]  : {"I":0, "like":1, "star":2 ...}
#받아온 시퀀스를 문장으로 원복시키자 word_index는 단어:숫자
#키와 밸류 바꿔치기 reverse_index → 숫자:단어
reverse_index = [(value, key) for (key, value) in word_index.items()]
for i in range(0, 10):
    print(reverse_index[i])
reverse_index = dict(reverse_index)
#dict으로 바꿔야 한번에 검색함

#id에 해당하는 train_data 시퀀스를 가져와 실제 문장으로 변환
#케라스만든 사람들이 0~3번까지는 특수목적으로 인덱스4부터 쓸모있음
def changeSentence(id):
    sequence = train_data[0]
    sentence = ' '.join(reverse_index.get(i-3, '*') for i in sequence) #⭐핵심변환로직
    #              합치자         워드인덱스에 3을 더해서저장  *:1만개만 가져오니까 없는단어가 있을경우 2번째인자로 출력. 잘보이라고 *표기
    # ' '.join():리스트로 생성된 단어들을 공백을 기준으로 하나의 문자열(문장)으로 합칩니다.
    print(sentence)
    # 실제 단어 인덱스 0, 1, 2, ...는 데이터셋에서는 3, 4, 5, ...로 저장됩니다. 따라서 원래 단어 인덱스를 얻으려면 i-3을 해야 합니다.
    # 0:padding, 1:start of sequence, 2:unknown word(어휘에 없는 단어)

#원하는대로 함수생성

print("--- Processing sentences ---")
#for문으로 다회호출가능
# changeSentence(0)
# changeSentence(1)
# for i in range(len(train_data)): # Loop through all available sequences in train_data
#     changeSentence(i)
# for 루프에서 changeSentence(i)가 train_data의 모든 리뷰에 대해 호출되기 때문에, 
# 훈련 데이터셋에 있는 25,000개의 영화 리뷰가 순차적으로 복원되어 출력되는 것입니다. 
# 따라서 "끝도 없이" 나오는 것처럼 보이는 것은 실제로 전체 훈련 데이터셋의 리뷰를 하나씩 출력하고 있기 때문입니다.
# 's life after all that was shared with us all
# * this film was just brilliant casting location scenery story direction everyone's really suited the part they played 
# and you could just imagine being there robert * is an amazing actor and now the same being director * father came 
# from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks 
# throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for * 
# and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad 
# and you know what they say if you cry at a film it must have been good and this definitely was also * to the 
# two little boy's that played the * of norman and paul they were just brilliant children are often left out of the * list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all

# *가 나오는 경우:
# num_words=10000 제한: IMDb 데이터셋을 로드할 때 imdb.load_data(num_words=10000)라고 지정했습니다. 이는 가장 자주 사용되는 단어 10000개만 가져오겠다는 의미입니다. 
# 따라서 만약 어떤 영화 리뷰에 이 10,000개 단어에 포함되지 않는 희귀하거나 빈도수가 낮은 단어가 있다면, 해당 단어는 reverse_index에 없을 것이고, 
# 이 경우 get(key, default_value) 메서드의 default_value인 *로 대체되어 출력됩니다.
# Keras 특수 인덱스: Keras의 IMDb 데이터셋은 0, 1, 2번 인덱스를 패딩, 문장 시작, 사전에 없는 단어(Out-of-Vocabulary, OOV) 등의 특수 목적으로 사용합니다. 
# 실제 단어는 3번 인덱스부터 시작하기 때문에 i-3을 해주는데, 만약 i-3의 결과가 reverse_index에 없는 값이라면 마찬가지로 *로 대체됩니다.
# 결론적으로, 'imdb1.py' 코드에서 '*'는 원본 영화 리뷰의 단어가 (자주 쓰이는 10,000개 단어 목록에 포함되지 않거나 Keras의 인덱스 규칙 때문에) reverse_index 딕셔너리에서 찾아지지 않을 때, 
# 그 자리를 채우는 대체 기호로 나옵니다. 이는 데이터의 손실이나 변형을 시각적으로 표시하기 위한 목적인 거죠.


#train_data : 시퀀스의 배열
#원핫인코딩 - 내부함수 말고 한번 만들어보자. 1만개까지만 불러오자
import numpy as np
def vectorize_sentences(sequences, dimensions=10000):
    # 문장 개수 * 10000개의 2D 배열을 0으로 채워 생성합니다.
    results = np.zeros((len(sequences), dimensions)) #zeros가 매개변수고 tuple받아감
    # 수정된 부분: 'sequences'를 순회하며 각 개별 시퀀스를 'sequence_data'에 할당합니다.
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
# [[0. 1. 1. ... 0. 0. 0.]
#  [0. 1. 1. ... 0. 0. 0.]
#  [0. 1. 1. ... 0. 0. 0.]]

#훈련셋과 검증셋 나눈다 전체데이터 25000, split함수써도됨
X_val = X_train[:10000] #검증셋10000개
X_train = X_train[10000:] #훈련셋은 15000개
y_val = train_labels[:10000]
y_train = train_labels[10000:]

#모델 생성
model = models.Sequential()
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#컴파일
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
model.compile( optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#모델 학습
# model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_val, y_val))
#히스토리 저장
history = model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_val, y_val))

#평가하기
print("훈련셋: ", model.evaluate(X_train, y_train))
print("테스트셋: ", model.evaluate(X_test, test_labels))

#예측
pred = model.predict(X_test)
print("첫 10개 예측 확률 배열:\n", pred[:10]) #라벨이 1이 될 확률 준다
def changeData(pred):
    for i in range(len(pred)):
        if pred[i] < 0.5:
            pred[i] = 0
        else:
            pred[i]
    return pred

pred = changeData(pred)
for i in range(0, 40):
    print(pred[i], test_labels[i])

# 1. 모델 학습 요약 (Epoch 50/50)
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Epoch 1/50
# 150/150 ━━━━━━━━━━━━━━━━━━━━ 5s 16ms/step - accuracy: 0.7718 - loss: 0.4991 - val_accuracy: 0.8883 - val_loss: 0.2828
# Epoch 2/50
# 150/150 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - accuracy: 0.9246 - loss: 0.2041 - val_accuracy: 0.8849 - val_loss: 0.2882
# ...
# Epoch 50/50
# 150/150 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - accuracy: 1.0000 - loss: 1.2379e-06 - val_accuracy: 0.8639 - val_loss: 1.7670
# • # 훈련데이터15000개이고 batch_size=100이므로, 15000 / 100 = 150개의 배치를 처리했다
# • 1s 8ms/step: 이에포크완료하는데 총1초걸렸고, 각스텝(배치처리)당 평균8밀리초가 소요되었다는 뜻.
# • accuracy: 1.0000: 훈련데이터셋에 대한 정확도100%라는 것을 의미. 모델이 훈련데이터를 완벽하게 학습했다는 뜻이죠.
# • loss: 1.3309e-06: 훈련데이터셋에 대한 손실(loss)값이 매우낮다는 것. 1.3309e-06은 $1.3309 \times 10^{-6}$으로, 거의 0에 가깝습니다. 
#        이는 모델이 훈련데이터에 대해 거의 완벽하게 예측하고 있음을 나타냅니다.
# • val_accuracy: 0.8636: 검증데이터셋(Validation dataset)에 대한 정확도가 약 86.36%라는 것을 의미. 훈련정확도(100%)와 비교했을 때 차이가 꽤 있습니다.
# • val_loss: 1.7318: 검증 데이터셋에 대한 손실 값입니다. 훈련 손실(거의 0)에 비해 상당히 높습니다.
# 주목할점: 훈련정확도와 손실은 매우 좋지만, 검증정확도와 손실은 상대적으로 많이 떨어집니다. 
#          모델이 과적합(Overfitting) 되었을 가능성이 높다는 신호입니다. 
#          즉, 모델이 훈련데이터의 특정패턴이나 노이즈까지 너무 외워서, 새데이터(검증셋)에 대해서 성능이 떨어지는 현상.

# 2. 최종 모델 평가
# 469/469 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 1.0000 - loss: 1.3708e-06    
# 훈련셋:  [1.2799672504115733e-06, 1.0]
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.8475 - loss: 1.9810    
# 테스트셋:  [1.93198561668396, 0.8502399921417236]
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step   

# 3. 예측 결과 (확률 및 이진 분류)
#  model.predict(X_test)를 통해 모델이 테스트 데이터에 대해 예측한 결과와, 그 예측값을 이진 분류(0 또는 1)로 변환한 후 실제 레이블(test_labels)과 비교한 것입니다.
# • [[5.1790499e-07], [1.0000000e+00], ...]:
# ㆍ이것은 model.predict(X_test)의 직접적인 출력입니다. 각 값은 해당 영화 리뷰가 긍정(클래스 1)일 확률을 나타냅니다.
# ㆍ예를 들어, 5.1790499e-07은 $5.179 \times 10^{-7}$로, 긍정일 확률이 거의 0에 가깝다는 뜻이므로, 모델은 이 리뷰를 부정(0)으로 예측할 것입니다.
# ㆍ1.0000000e+00은 확률이 100%이므로 긍정(1)으로 예측할 것입니다.
# • [0.] 0, [1.] 1, [1.] 1, [0.9999993] 0, ...`:
# ㆍ이 부분은 changeData 함수를 거쳐 확률이 이진 값(0 또는 1)으로 변환된 모델의 최종 예측값과 그 옆에 실제 정답 레이블이 나란히 출력된 것입니다.
# ㆍ[0.] 0: 모델이 0으로 예측했고, 실제 정답도 0입니다 (정확한 예측).
# ㆍ[1.] 1: 모델이 1로 예측했고, 실제 정답도 1입니다 (정확한 예측).
# ㆍ[0.9999993] 0: 모델이 0.9999993이라는 높은 확률을 출력했고, 이를 changeData 함수가 1로 변환했을 것입니다. 
#            하지만 실제 정답은 0입니다. 이는 모델이 **오분류(misclassification)**한 사례입니다. 
#   이 경우처럼 모델이 긍정으로 강하게 예측했지만 실제로는 부정인 경우가 나타나는 것을 볼 수 있습니다.
# 결론: 어떤 확률을 예측했는지, 이 확률기준으로 최종적으로 어떤감성(긍정/부정)예측했는지, 실제정답과 일치하는지를 보여줍니다. 
#      여기서도 모델의 오류(오분류) 사례들을 직접 확인.
# [[5.1790499e-07]
#  [1.0000000e+00]
#  [1.0000000e+00]
#  [9.9999928e-01]
#  [9.9999988e-01]
#  [9.9999970e-01]
#  [1.0000000e+00]
#  [1.8037610e-12]
#  [9.9999994e-01]
#  [1.0000000e+00]]
# [0.] 0    모델이 0으로 예측했고, 실제 정답도 0(정확한예측)
# [1.] 1
# [1.] 1
# [0.9999993] 0
# [0.9999999] 1
# [0.9999997] 1
# [1.] 1
# [0.] 0
# [0.99999994] 0
# [1.] 1
# [0.8327918] 1
# [0.] 0
# [0.] 0
# [0.] 0
# [1.] 1
# [0.] 0
# [0.] 1
# [0.] 0
# [0.] 0
# [0.] 0
# [1.] 1
# [1.] 1
# [0.] 1
# [1.] 1
# [0.99999994] 1
# [1.] 1
# [0.] 0
# [0.99998784] 1
# [1.] 1
# [0.] 0
# [1.] 1
# [0.] 1
# [0.99999946] 0
# [0.] 0
# [0.] 0
# [0.] 0
# [1.] 1
# [1.] 1
# [0.] 0
# [0.] 0

#훈련과 검증 정확도 그리기
import matplotlib.pyplot as plt
history_dict = history.history
acc = history_dict['accuracy'] #훈련셋 정확도
val_acc = history_dict['val_accuracy'] #검증셋 정확도

length = range(1, len(history_dict['accuracy'])+1 ) #X축만들기
plt.plot(length, acc, 'bo', label='Training acc')
plt.plot(length, val_acc, 'b', label='Validation acc')
plt.title("IMDB Training and Validation acc")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

# 전체적인 시사점
# 모델이 훈련데이터에는 거의 완벽하게 적합되었지만, 새데이터(검증셋,테스트셋)에 대해서는 성능이 눈에띄게 떨어지는 전형적인 과적합을 보여줍니다.
# 감성 분류 모델을 실제 서비스에 적용하려면 이 과적합 문제를 해결해야 합니다.

# 개선 방안으로 고려할 수 있는 것들:
# • 에포크 수 줄이기: 현재 50 에포크는 너무 많을 수 있습니다. 검증 손실이 증가하기 시작하는 지점(일반적으로 '조기 종료'라고 함)에서 학습을 멈추는 것이 좋습니다.
# • 모델 복잡도 줄이기: 레이어의 뉴런 수를 줄이거나, 레이어 개수를 줄여 모델을 더 단순하게 만듭니다.
# • 드롭아웃(Dropout) 사용: 학습 시 특정 뉴런을 임의로 끄는 기법으로 과적합을 방지하는 데 효과적입니다.
# • 데이터 증강(Data Augmentation): 훈련 데이터의 양을 늘리거나 다양성을 추가합니다 (텍스트 데이터에서는 동의어 교체, 문장 재구성 등으로 가능).
# • 정규화(Regularization): L1 또는 L2 정규화와 같은 기법을 사용하여 모델 가중치가 너무 커지는 것을 방지합니다.


# 코드흐름
# 1.데이터 로드: imdb.load_data(num_words=10000)를 사용해 영화 리뷰 데이터와 해당 레이블을 불러옵니다. train_data, train_labels, test_data, test_labels로 나뉩니다.
# 2.데이터 형태 확인: 불러온 데이터의 크기(shape)를 출력하여 데이터의 구조를 파악합니다.
# 3.원시 데이터 확인: train_data[:3]을 출력하여 실제 리뷰 데이터가 정수 리스트 형태로 어떻게 구성되어 있는지 보여줍니다. train_labels[:3]은 해당 리뷰의 감성 레이블을 보여줍니다.

# 4.단어 인덱스 매핑:
# • `imdb.get_word_index()`를 통해 단어(문자열)와 해당 단어의 인덱스(정수)가 매핑된 딕셔너리(word_index)를 가져옵니다.
# • `showDictionary(10)` 함수를 통해 `word_index`의 몇 가지 예시를 출력하여 단어-인덱스 매핑을 보여줍니다.
# • `word_index`를 뒤집어 인덱스(정수)와 해당 인덱스에 매핑된 단어(문자열)가 매핑된 딕셔너리(`reverse_index`)를 생성합니다. 이는 정수 시퀀스를 원래 문장으로 복원하는 데 사용됩니다.

# 5.문장 복원 (`changeSentence` 함수):
# • `changeSentence(id)` 함수는 `train_data`에서 특정 `id`에 해당하는 정수 시퀀스를 가져옵니다.
# • 이 시퀀스의 각 정수 i에 대해 `reverse_index.get(i-3, '*')`를 사용하여 실제 단어를 찾습니다. i-3을 하는 이유는 Keras의 IMDb 데이터셋에서 0, 1, 2번 인덱스가 특수 문자로 예약되어 있기 때문입니다. 
#   만약 해당 인덱스에 매핑되는 단어를 찾을 수 없으면 *로 표시합니다 (이는 `num_words` 제한 때문에 발생할 수 있습니다).
# • 이렇게 찾은 단어들을 공백으로 연결하여 하나의 문장으로 만들고 출력합니다.

# 6.전체 문장 출력: for 루프를 사용하여 `train_data`에 있는 모든 영화 리뷰 시퀀스를 하나씩 `changeSentence` 함수에 넣어 원래 문장으로 복원하여 출력합니다.