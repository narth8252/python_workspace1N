#250801 PM1시 쌤PPT 딥러닝종합_백현숙.ppt 465p 단어임베딩학습
# conda install gensim (conda activate mytensorfolw, deeplearning)
# C:\ProgramData\anaconda3\envs 에서 각 가상환경폴더 우클릭>속성
# deeplearning가상환경에서 권한이없다고 failed해서 폴더찾아 속성>보안>system>편집 에 쓰기권한허용

#중요: konlpy하고 Korpora 는 pip 로 설치해야 한다 
#https://fasttext.cc/docs/en/crawl-vectors.html
""" 
# Word2Vec 다운받기
# Direct Link (S3 AWS):
#  https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
# Hugging Face: 최근에는 Hugging Face datasets나 models에서도 찾아볼 수 있습니다.
#  예를 들어, NathaNn1111/word2vec-google-news-negative-300-bin와 같은 저장소에서 다운로드 링크를 제공합니다.
# Kaggle: Kaggle에도 이 데이터셋이 업로드되어 있습니다.
#  GoogleNews-vectors-negative300.bin.gz를 검색하면 찾을 수 있습니다.
#  https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

영화 리뷰 텍스트 데이터셋(IMDb)을 사용하여 감성 분류(긍정/부정) 모델을 구축하고 훈련하는 과정을 담고 있습니다. 
특히, 사전 학습된 Word2Vec 임베딩을 활용하여 모델의 성능을 향상시키는 데 초점을 맞추고 있습니다.
"""

import keras.initializers
import requests
import subprocess
import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import os, pathlib, shutil, random
from tensorflow.keras.utils import text_dataset_from_directory

# 1.데이터준비 및 전처리
# 1-1)라벨링 
def labeling(): 
    base_dir = pathlib.Path("data/aclImdb_new") 
    val_dir = base_dir/"val"   # pathlib 객체에  / "디렉토리" => 결과가 문자열이 아니다 
    train_dir = base_dir/"train"

    for category in ("neg", "pos"):
        os.makedirs(val_dir/category)  #디렉토리를 만들고 
        files = os.listdir(train_dir/category) #해당 카테고리의 파일 목록을 모두 가져온다 
        random.Random(1337).shuffle(files) #파일을 랜덤하게 섞어서 복사하려고 파일 목록을 모두 섞는다 
        num_val_samples = int(0.2 * len(files))  #20%를 검증 데이터로 분리하여 data/aclImdb_new 디렉토리 구조를 생성
        val_files = files[-num_val_samples:] #20%만 val폴더로 이동한다 
        for fname in val_files:
            shutil.move(train_dir/category/fname, val_dir/category/fname )    
# labeling() #이 함수는 한 번 실행 후 주석 처리

# 1-2)Keras 사용한 데이터로드
#데이터셋을 활용해서 디렉토리로부터 파일을 불러와서 벡터화를 진행한다
# train,val,test세트를 각각 tf.data.Dataset 객체로 로드.(대용량 데이터 처리 및 파이프라인 구성에 효율적)
import keras
batch_size = 32 #한번에 읽어올 양
train_ds = keras.utils.text_dataset_from_directory(
    "data/aclImdb_new/train", #디렉토리명
    batch_size=batch_size
)

val_ds = keras.utils.text_dataset_from_directory(
    "data/aclImdb_new/val", #디렉토리명
    batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "data/aclImdb_new/test", #디렉토리명
    batch_size=batch_size
)

#데이터셋은 inputs,targets 반복해서 갖고오지만 필요한건inputs
for inputs, targets in train_ds: #실제 읽어오는 데이터 확인
    print("inputs.shape", inputs.shape)
    print("inputs.dtype", inputs.dtype)
    print("targets.shape", targets.shape)
    print("targets.dtype", targets.dtype)
    print("inputs[0]", inputs[:3])
    print("targets[0]", targets[:3])
    break #하나만 출력해보자
    #0부정, 1긍정 → 폴더명 정렬해서 0,1,2 이런식으로 라벨링 neg-0 pos-1)

# 1-3)TextVectorization 레이어를 통한 텍스트 벡터화
#시퀀스 생성
max_length = 600 #한평론에서 사용하는 단어의 최대길이
max_tokens = 20000 #고빈도사용단어 수

# TextVectorization 레이어를 통한 텍스트 벡터화:
text_vectorization = TextVectorization(
    max_tokens = max_tokens,
    output_mode = "int", #텍스트를 정수 시퀀스(인덱스)로 변환
    output_sequence_length = max_length #단어임베딩층 사용하려면 반드시 모든 시퀀스길이 고정
)

# adapt() 메서드를 사용하여 훈련 데이터에서 어휘 사전을 구축
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds) #어휘사전 만들어야한다

# map() 함수를 사용하여 원본 텍스트 데이터셋을 정수 인덱스 시퀀스로 변환
int_train_ds = train_ds.map(lambda x, y:(text_vectorization(x), y), num_parallel_calls=1)
int_val_ds = val_ds.map(lambda x, y:(text_vectorization(x), y), num_parallel_calls=1)
int_test_ds = test_ds.map(lambda x, y:(text_vectorization(x), y), num_parallel_calls=1)

#임베딩 내부구조 엿보기
for item in int_train_ds:
    print(item)
    break

# 2.사전 학습된 Word2Vec 임베딩 로드 및 준비: 
# 2-1)Word2Vec 모델 불러오기
from gensim.models import KeyedVectors
filename = "./data/GoogleNews-vectors-negative300.bin"
# 이 파일은 방대한 양의 텍스트 데이터로 미리 학습된 단어 임베딩을 포함

try: #임베딩 차원 확인
    word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)
    #파일이 txt형태, binary형태 있는데, binary형태를 읽겠다
    embedding_dim = word2vec_model.vector_size
except FileNotFoundError:
    print(filename + "을 찾을 수 없습니다.")
    exit() #프로그램종료
except Exception as e:
    print("에러발생" + e)
    exit() #프로그램종료
print("파일 로딩 성공")


# # 사전학습된 임베딩 데이터 불러오기
# # https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt?select=glove.6B.100d.txt
# #임베딩데이터, 단어별 각 단어와의 거리가 벡터로저장, 파일명의 100이 출력벡터크기

# # path_to_glove_file = "./Deep_Learning/data/glove.6B.100d.txt"
# path_to_glove_file = "./data/glove.6B.100d.txt"
# embedding_index = {}
# with open(path_to_glove_file, encoding="utf-8") as f:
#     for line in f: 
#         word, coefs = line.split(maxsplit=1)
#         #단어, 단어들간의 벡터구조로 돼있다. 예)the, 0.0012, 000172
#         coefs = np.fromstring(coefs, "f", sep=" ") #나머지벡터들을 numpy배열로 전환
#         embedding_index[word] = coefs
#         # print(embedding_index) #궁금하니까 1개만 출력후 break
#         # break
# #{"the":[ , , , ,]}
# print("개수", len(embedding_index))
# #print(embeddings_index)

# 2-3)임베딩 행렬 구축: 우리데이터와 연동
vocabulary = text_vectorization.get_vocabulary() #text_vectorization 레이어에서 구축한 우리데이터의 어휘사전 가져오기

word_index = dict(zip(vocabulary, range(len(vocabulary))))
# embedding_dim = 100 #미리학습한 임베딩층 출력값100개
embedding_matrix = np.zeros((max_tokens, embedding_dim))
# max_tokens 크기와 embedding_dim 크기를 가진 embedding_matrix를 0으로 초기화
#케라스 Embedding레이어의 초기값
# embedding_matrix를 embedding_index 정보로 채워야함: 우리데이터의 어휘사전 단어에 대해 Word2Vec 모델에서 해당단어의 벡터찾아 embedding_matrix에 채워 넣습니다. 모델에 없으면 해당단어벡터는 0 유지(KeyError 발생 시 pass).

# word_index는 {단어:인덱스}형태임
for word, i in word_index.items():
    #단어와 인덱스 가져온다
    if i < max_tokens: #혹시나 20000개를 넘어가는 토큰있을까봐 오류처리
        try:
            # 올바른 방법: 딕셔너리처럼 접근하거나 get_vector() 메서드 사용
            embedding_vector = word2vec_model[word] # 단어를 키로 사용하여 벡터에 접근
            # 또는 embedding_vector = word2vec_model.get_vector(word)
            embedding_matrix[i] = embedding_vector
        except KeyError:
            pass # 건너뛰기

print(embedding_matrix[:10])

# 3. 모델구축(Keras Functional API)
from keras import models, layers
import tensorflow as tf
# 3-1)입력 레이어
inputs = keras.Input(shape=(None,), dtype="int64")
#3-2)사전학습된 임베딩 레이어 사용
embedded = layers.Embedding(
    input_dim = max_tokens, #input_dim=어휘사전크기로 설정
    output_dim=embedding_dim, #output_dim=Word2Vec 벡터크기로 설정
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    ####이부분 필수. 사전학습된 층에 바꿔치기(미리 준비한 embedding_matrix로 임베딩레이어의 가중치 초기화)
    trainable=False, ###임베딩가중치를 훈련중에 업데이트 할거냐? 사전학습된 인베딩층 사용할때 False(사전학습된 임베딩은 훈련되지않도록 설정)
    mask_zero=True #패딩토큰(0)을 마스킹하여 LSTM이 이를 무시하도록 함
    )(inputs)
print(embedded.shape)

#양방향 RNN
#          양방향감싸서 약방향처리     32개 유닛가진 LSTM층, 시퀀스데이터에서 장기의존성 포착하는데 사용
# NLP에서 Bi-LSTM이 유익한 이유
# LSTM 레이어를 감싸서 사실상 두 개의 독립적인 LSTM(하나는 정방향, 하나는 역방향)을 생성하고 그들의 출력을 연결합니다. 각 LSTM이 32개의 유닛을 가지므로, Bidirectional(LSTM(32)) 레이어의 연결된 출력은 32 + 32 = 64개의 유닛을 가지게 됩니다(여러분의 model.summary() 출력에서 (None, 64)로 확인됩니다).
# 예시: "The bank of the river was muddy." 에서 "bank"를 처리할 때: 
# 정방향 LSTM은 "The"로부터 문맥을 가져옵니다.
# 역방향 LSTM("muddy"에서 "The"로 거슬러 처리)은 "of the river"로부터 문맥을 가져옵니다.
# 이들을 결합함으로써 Bi-LSTM은 미래 정보가 전파되기를 기다릴 필요 없이 "bank" 시간 단계에서 "강둑(river bank)"으로서의 "bank"에 대한 더 정확한 표현을 형성할 수 있습니다.

#3-3) 양방향 LSTM레이어 
# x = Bidirectional(LSTM(units=32, return_sequences=False))(embedded)
# `return_sequences=False`는 마지막 은닉 상태(또는 연결된 마지막 상태)만 반환한다는 의미입니다.
# 다른 순환 레이어를 쌓으려면 `True`로 설정합니다.
x = layers.Bidirectional(layers.LSTM(32))(embedded) #임베딩된 텐서를 LSTM 입력으로 사용
x = layers.Dropout(0.5)(x) #과적합방지위해 50%드롭아웃뉴런을 랜덤하게 비활성화)
outputs = layers.Dense(1, activation='sigmoid')(x) #최종출력층( 이진 분류)
model = keras.Model(inputs, outputs)    #모델정의
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
model.summary()

# 4. 모델훈련 및 평가
#4-1)저장:훈련 중 검증 손실이 가장 낮은 모델을 RNN1.keras 파일로 저장합니다.
callbacks = [
    keras.callbacks.ModelCheckpoint("RNN1.keras", save_best_only=True)
]

#4-2)model.fit()을 사용하여 int_train_ds로 모델을 훈련하고, int_val_ds로 검증
model.fit(int_train_ds, validation_data=int_val_ds, epochs=3, callbacks=callbacks)
print("테스트셋", model.evaluate(int_test_ds))
#4-3)훈련완료후 model.evaluate()사용해 int_test_ds에서 모델성능(손실 및 정확도) 최종평가

# 실행 결과 분석:
# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Deep_Learning>python 0801NLP_Embedding3.py
# 2025-08-01 13:50:33.415530: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-08-01 13:50:36.392888: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found 20000 files belonging to 2 classes. #데이터셋로드(훈련,검증,테스트 파일수가 정확히 로드됨)
# 2025-08-01 13:50:49.019225: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 5000 files belonging to 2 classes.  #데이터셋로드(훈련,검증,테스트 파일수가 정확히 로드됨)
# Found 23215 files belonging to 2 classes. #데이터셋로드(훈련,검증,테스트 파일수가 정확히 로드됨)
# 2025-08-01 13:51:31.510233: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
# (<tf.Tensor: shape=(32, 600), dtype=int64, numpy=
# array([[8823,  256,   74, ...,    0,    0,    0],
#        [  74,   14, 1994, ...,    0,    0,    0],
#        [  10,  153,   54, ...,    0,    0,    0],
#        ...,
#        [   8,    4,   22, ...,    0,    0,    0],
#        [  10,  237,   53, ...,    0,    0,    0],
#        [  44, 4197, 8248, ...,    0,    0,    0]], dtype=int64)>, <tf.Tensor: shape=(32,), dtype=int32, numpy=
# array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1,
#        1, 1, 0, 0, 0, 1, 0, 1, 0, 0])>)
# [[ 0.          0.          0.         ...  0.          0.
#    0.        ]
#  [ 0.          0.          0.         ...  0.          0.
#    0.        ]
#  [ 0.08007812  0.10498047  0.04980469 ...  0.00366211  0.04760742
#   -0.06884766]
#  ...
#  [ 0.00704956 -0.07324219  0.171875   ...  0.01123047  0.1640625
#    0.10693359]
#  [ 0.0703125   0.08691406  0.08789062 ... -0.04760742  0.01446533
#   -0.0625    ]
#  [ 0.08447266 -0.00035286  0.05322266 ...  0.01708984  0.06079102
#   -0.10888672]]
# (None, None, 300)
# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)      │ (None, None)              │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ embedding (Embedding)         │ (None, None, 300)         │       6,000,000 │ input_layer[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ not_equal (NotEqual)          │ (None, None)              │               0 │ input_layer[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ bidirectional (Bidirectional) │ (None, 64)                │          85,248 │ embedding[0][0],           │
# │                               │                           │                 │ not_equal[0][0]            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dropout (Dropout)             │ (None, 64)                │               0 │ bidirectional[0][0]        │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense (Dense)                 │ (None, 1)                 │              65 │ dropout[0][0]              │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 6,085,313 (23.21 MB)
#  Trainable params: 85,313 (333.25 KB)
#  Non-trainable params: 6,000,000 (22.89 MB)
# Epoch 1/2
# 2025-08-01 13:52:25.403654: E tensorflow/core/util/util.cc:131] oneDNN supports DT_BOOL only on platforms with AVX-512. Falling back to the default Eigen-based implementation if present.
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 304s 477ms/step - accuracy: 0.6851 - loss: 0.5707 - val_accuracy: 0.8148 - val_loss: 0.4090
# Epoch 2/2
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 242s 386ms/step - accuracy: 0.8090 - loss: 0.4287 - val_accuracy: 0.7798 - val_loss: 0.4418
# 726/726 ━━━━━━━━━━━━━━━━━━━━ 166s 228ms/step - accuracy: 0.7773 - loss: 0.4473
# 테스트셋 [0.45122796297073364, 0.7727331519126892]

""" 
# 제공해주신 코드는 영화 리뷰 텍스트 데이터셋(IMDb)을 사용하여 감성 분류(긍정/부정) 모델을 구축하고 훈련하는 과정을 담고 있습니다. 특히, 사전 학습된 Word2Vec 임베딩을 활용하여 모델의 성능을 향상시키는 데 초점을 맞추고 있습니다.
# 코드의 전체적인 흐름은 다음과 같습니다:

# 데이터 준비 및 전처리:
# 데이터셋 다운로드 및 구성: aclImdb 데이터셋을 사용하며, labeling() 함수를 통해 훈련 데이터의 20%를 검증 데이터로 분리하여 data/aclImdb_new 디렉토리 구조를 생성합니다. (이 함수는 한 번 실행 후 주석 처리해야 합니다.)
# Keras text_dataset_from_directory를 사용한 데이터 로드: train, val, test 세트를 각각 tf.data.Dataset 객체로 로드합니다. 이는 대용량 데이터 처리 및 파이프라인 구성에 효율적입니다.

# TextVectorization 레이어를 통한 텍스트 벡터화:
# 최대 max_tokens(20000개)의 고빈도 단어만 사용하도록 설정합니다.
# 출력 모드를 "int"로 설정하여 각 단어를 정수 인덱스로 변환합니다.
# output_sequence_length를 max_length(600)로 설정하여 모든 시퀀스의 길이를 고정합니다. 이는 임베딩 레이어의 입력으로 필요합니다.
# adapt() 메서드를 사용하여 훈련 데이터에서 어휘 사전을 구축합니다.
# map() 함수를 사용하여 원본 텍스트 데이터셋을 정수 인덱스 시퀀스로 변환합니다.

# 사전 학습된 Word2Vec 임베딩 로드 및 준비:
# Word2Vec 모델 로드: gensim.models.KeyedVectors.load_word2vec_format()을 사용하여 GoogleNews-vectors-negative300.bin 파일을 로드합니다. 이 파일은 방대한 양의 텍스트 데이터로 미리 학습된 단어 임베딩을 포함하고 있습니다.
# 임베딩 차원 확인: 로드된 Word2Vec 모델의 벡터 크기(word2vec_model.vector_size, 여기서는 300)를 embedding_dim으로 설정합니다.

# 임베딩 행렬 구축:
# text_vectorization 레이어에서 구축한 우리 데이터의 어휘 사전을 가져옵니다.
# max_tokens 크기와 embedding_dim 크기를 가진 embedding_matrix를 0으로 초기화합니다.
# 우리 데이터의 어휘 사전에 있는 각 단어에 대해 Word2Vec 모델에서 해당 단어의 벡터를 찾아 embedding_matrix에 채워 넣습니다. 만약 Word2Vec 모델에 없는 단어라면 해당 단어의 벡터는 0으로 유지됩니다 (KeyError 발생 시 pass).

# 모델 구축 (Keras Functional API):
# 입력 레이어: max_length (실제로는 None으로 시퀀스 길이에 유연하게 대응)의 정수 시퀀스를 입력으로 받습니다.
# 임베딩 레이어:
# input_dim은 max_tokens(어휘 사전 크기)로 설정합니다.
# output_dim은 embedding_dim(Word2Vec 벡터 크기)으로 설정합니다.
# 핵심: embeddings_initializer=keras.initializers.Constant(embedding_matrix)를 사용하여 미리 준비한 embedding_matrix로 임베딩 레이어의 가중치를 초기화합니다.
# 핵심: trainable=False로 설정하여 이 임베딩 레이어의 가중치가 훈련 중에 업데이트되지 않도록 합니다. 이는 사전 학습된 임베딩의 의미를 보존하기 위함입니다.
# mask_zero=True는 패딩 토큰(0)을 마스킹하여 LSTM이 패딩을 실제 시퀀스의 일부로 처리하지 않도록 합니다.

# 양방향 LSTM 레이어: layers.Bidirectional(layers.LSTM(32))를 사용하여 임베딩된 시퀀스를 처리합니다. 양방향 LSTM은 텍스트의 앞뒤 문맥 정보를 모두 활용하여 단어의 의미를 더 잘 이해할 수 있도록 돕습니다. 32개의 LSTM 유닛을 사용합니다.
# 드롭아웃 레이어: layers.Dropout(0.5)를 적용하여 과적합을 방지합니다. 50%의 뉴런을 랜덤하게 비활성화합니다.
# 출력 레이어: layers.Dense(1, activation='sigmoid')를 사용하여 이진 분류(긍정/부정)를 위한 최종 출력을 생성합니다. sigmoid 활성화 함수는 0과 1 사이의 확률 값을 출력합니다.
# 모델 컴파일: rmsprop 옵티마이저, binary_crossentropy 손실 함수(이진 분류), accuracy 지표를 사용하여 모델을 컴파일합니다.

# 모델 훈련 및 평가:
# 콜백 설정: ModelCheckpoint 콜백을 사용하여 훈련 중 검증 손실이 가장 낮은 모델을 RNN1.keras 파일로 저장합니다.
# 모델 훈련: model.fit()을 사용하여 int_train_ds로 모델을 훈련하고, int_val_ds로 검증합니다. epochs=3으로 설정되어 3번의 훈련 에포크를 실행합니다.
# 모델 평가: 훈련이 완료된 후 model.evaluate()를 사용하여 int_test_ds에서 모델의 성능(손실 및 정확도)을 최종적으로 평가합니다.

### 실행 결과 분석:
# 데이터셋 로드: 훈련, 검증, 테스트 파일 수가 정확히 로드되었음을 보여줍니다. (Found 20000 files belonging to 2 classes., Found 5000 files belonging to 2 classes., Found 23215 files belonging to 2 classes.)
# Word2Vec 로딩 성공: KeyedVectors.load_word2vec_format에서 파일이 성공적으로 로드되었음을 추론할 수 있습니다.
# 임베딩 매트릭스 출력: embedding_matrix[:10]는 Word2Vec 모델에서 가져온 단어 임베딩의 일부를 보여줍니다. 첫 두 줄이 0인 것은, max_tokens에 해당하는 단어 중 Word2Vec 모델에 없는 단어이거나, 패딩 토큰(0번 인덱스)에 해당하기 때문일 수 있습니다.
# 모델 요약 (model.summary()):
# Embedding 레이어의 파라미터 수가 6,000,000 (max_tokens * embedding_dim = 20000 * 300)이고 Non-trainable params에 포함되어 있어, 임베딩 가중치가 훈련되지 않음을 명확히 보여줍니다.
# Bidirectional (LSTM) 레이어와 Dense 레이어의 파라미터만 Trainable params로 계산되어 있습니다.

# 훈련 과정:
# 에포크별 accuracy와 loss가 출력됩니다. val_accuracy와 val_loss는 검증 세트에서의 성능을 보여줍니다.
# 첫 에포크에서 accuracy: 0.6851 -> val_accuracy: 0.8148로 검증 정확도가 상당히 높게 나왔습니다. 이는 사전 학습된 임베딩의 효과일 수 있습니다.
# 두 번째 에포크에서는 훈련 정확도는 0.8090으로 올랐지만, 검증 정확도는 0.7798로 약간 떨어졌습니다. 이는 과적합의 초기 징후일 수 있습니다.
# 테스트셋 평가: 최종적으로 int_test_ds에서 모델을 평가한 결과 [손실, 정확도] 형태로 출력됩니다. [0.45122796297073364, 0.7727331519126892]로, 약 **77.27%**의 테스트 정확도를 보였습니다.

### 전반적으로 이 코드는 Word2Vec과 Bidirectional LSTM을 활용한 효과적인 텍스트 감성 분류 모델 구축 과정을 잘 보여주고 있습니다.
"""