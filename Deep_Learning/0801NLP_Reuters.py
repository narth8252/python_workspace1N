#250801 PM1시 쌤PPT 딥러닝종합_백현숙.ppt 465p 단어임베딩학습
# conda install gensim (conda activate mytensorfolw, deeplearning)
# C:\ProgramData\anaconda3\envs 에서 각 가상환경폴더 우클릭>속성
# deeplearning가상환경에서 권한이없다고 failed해서 폴더찾아 속성>보안>system>편집 에 쓰기권한허용
#중요: konlpy하고 Korpora 는 pip 로 설치해야 한다 
#https://fasttext.cc/docs/en/crawl-vectors.html
""" 
# 로이터 다운받기
https://www.kaggle.com/datasets/datajameson/reuters-news-dataset
"""
#https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/nsmc.html

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from konlpy.tag import Okt
import re
import os, pathlib, shutil, random
import numpy as np # Used for dummy dataset creation
from keras import layers
import requests
from gensim.models import KeyedVectors

#라벨링 
def labeling(): 
    base_dir = pathlib.Path("data/reuters") 
    val_dir = base_dir/"val"   # pathlib 객체에  / "디렉토리" => 결과가 문자열이 아니다 
    train_dir = base_dir/"train"

    for category in ("neg", "pos"):
        os.makedirs(val_dir/category)  #디렉토리를 만들고 
        files = os.listdir(train_dir/category) #해당 카테고리의 파일 목록을 모두 가져온다 
        random.Random(1337).shuffle(files) #파일을 랜덤하게 섞어서 복사하려고 파일 목록을 모두 섞는다 
        num_val_samples = int(0.2 * len(files)) 
        val_files = files[-num_val_samples:] #20%만 val폴더로 이동한다 
        for fname in val_files:
            shutil.move(train_dir/category/fname, val_dir/category/fname )    
# labeling() #하고 주석처리

#데이터셋을 활용해서 디렉토리로부터 파일을 불러와서 벡터화를 진행한다 
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
    # print("inputs.shape", inputs.shape)
    # print("inputs.dtype", inputs.dtype)
    # print("targets.shape", targets.shape)
    # print("targets.dtype", targets.dtype)
    # print("inputs[0]", inputs[:3])
    # print("targets[0]", targets[:3])
    break #하나만 출력해보자
    #0부정, 1긍정 → 폴더명 정렬해서 0,1,2 이런식으로 라벨링 neg-0 pos-1)

#시퀀스 생성
max_length = 600 #한평론에서 사용하는 단어의 최대길이
max_tokens = 20000 #고빈도사용단어 수

# TextVectorization 레이어를 통한 텍스트 벡터화:
text_vectorization = TextVectorization(
    max_tokens = max_tokens,
    output_mode = "int", #텍스트를 정수 시퀀스로 변환
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

#Word23Vec 파일 불러오기
from gensim.models import KeyedVectors
filename = "./data/GoogleNews-vectors-negative300.bin"

try:
    word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)
    #파일이 txt형태, binary형태 있는데, binary형태를 읽겠다
    embedding_dim = word2vec_model.vector_size
except FileNotFoundError:
    print(filename + "을 찾을 수 없습니다.")
    exit() #프로그램종료
except Exception as e:
    print("에러발생" + e)
    exit() #프로그램종료
# print("파일 로딩 성공")


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

#우리데이터와 연동
vocabulary = text_vectorization.get_vocabulary() #우리 어휘사전 가져오기

word_index = dict(zip(vocabulary, range(len(vocabulary))))
# embedding_dim = 100 #미리학습한 임베딩층 출력값100개
embedding_matrix = np.zeros((max_tokens, embedding_dim))
#케라스 Embedding레이어의 초기값
# embedding_matrix를 embedding_index 정보로 채워야한다.

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

from keras import models, layers
import tensorflow as tf

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(
    input_dim = max_tokens, 
    output_dim=embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    ####이부분 필수. 사전학습된 층에 바꿔치기
    trainable=False, #임베딩가중치를 훈련중에 업데이트 할거냐? 사전학습된 인베딩층 사용할때 False
    mask_zero=True
    )(inputs)
print(embedded.shape)

#양방향 RNN
#          양방향감싸서 약방향처리     32개 유닛가진 LSTM층, 시퀀스데이터에서 장기의존성 포착하는데 사용
# LSTM 레이어를 감싸서 사실상 두 개의 독립적인 LSTM(하나는 정방향, 하나는 역방향)을 생성하고 그들의 출력을 연결합니다. 각 LSTM이 32개의 유닛을 가지므로, Bidirectional(LSTM(32)) 레이어의 연결된 출력은 32 + 32 = 64개의 유닛을 가지게 됩니다(여러분의 model.summary() 출력에서 (None, 64)로 확인됩니다).
# 예시: "The bank of the river was muddy." 에서 "bank"를 처리할 때: 
# 정방향 LSTM은 "The"로부터 문맥을 가져옵니다.
# 역방향 LSTM("muddy"에서 "The"로 거슬러 처리)은 "of the river"로부터 문맥을 가져옵니다.
# 이들을 결합함으로써 Bi-LSTM은 미래 정보가 전파되기를 기다릴 필요 없이 "bank" 시간 단계에서 "강둑(river bank)"으로서의 "bank"에 대한 더 정확한 표현을 형성할 수 있습니다.

x = layers.Bidirectional(layers.LSTM(32))(embedded) #임베딩된 텐서를 LSTM 입력으로 사용
x = layers.Dropout(0.5)(x) #과적합방지위해 50%드롭아웃 적용
outputs = layers.Dense(1, activation='sigmoid')(x) #최종출력층
model = keras.Model(inputs, outputs)    #모델정의
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
model.summary()

#저장
callbacks = [
    keras.callbacks.ModelCheckpoint("RNN1.keras", save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=3, callbacks=callbacks)
print("테스트셋", model.evaluate(int_test_ds))



# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Deep_Learning>python 0801NLP_Embedding3.py
# 2025-08-01 13:50:33.415530: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-08-01 13:50:36.392888: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found 20000 files belonging to 2 classes.
# 2025-08-01 13:50:49.019225: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 5000 files belonging to 2 classes.
# Found 23215 files belonging to 2 classes.
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
