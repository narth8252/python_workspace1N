#250731 PM3시 쌤PPT 딥러닝종합_백현숙.ppt 465p RNN 단어임베딩학습
import keras.initializers
import requests
import subprocess
import re
import string
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import os, pathlib, shutil, random
from tensorflow.keras.utils import text_dataset_from_directory

#데이터 다운로드 g후 주석처리
# def download():
#     url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#     file_name = "aclImdb_v1.tar.gz"

#     response = requests.get(url, stream=True)  #스트리밍 방식 다운로드
#     with open(file_name, "wb") as file:
#         for chunk in response.iter_content(chunk_size=8192):  #8KB씩 다운로드
#             file.write(chunk)
#     print("Download complete!")
# download() #파일다운로드: 용량너무커서 8192개씩 잘라서 저장하는 코드

#압축풀기 후 주석처리 : 프로그램호출  → 프로세스,tar라이브러리 있어야함 (반디집으로 풀어도됨.알집X)
# def release():
#     subprocess.run(["tar", "-xvzf", "aclImdb_v1.tar.gz"], shell=True) #tar 프로그램 가동하기 
#     #tar.gz => linux에서는 파일을 여러개를 한번에 압축을 못함 tar라는 형식으로 압축할 모든 파일을 하나로 묶어서 패키지로 만든다음에 
#     #          압축을 한다.  tar , gz가동  그래서 압축풀고 다시 패키지도 풀어야 한다. 
#     #          tar  -xvzf 파일명   형태임         
#     print("압축풀기 완료")
#release() #압출풀고 주석처리

#train 폴더에 unsup폴더 삭제해야함(분석에 불필요, 라벨이 2개여야함)

#라벨링 
def labeling(): 
    base_dir = pathlib.Path("data/aclImdb_new") 
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
    print("inputs.shape", inputs.shape)
    print("inputs.dtype", inputs.dtype)
    print("targets.shape", targets.shape)
    print("targets.dtype", targets.dtype)
    print("inputs[0]", inputs[:3])
    print("targets[0]", targets[:3])
    break #하나만 출력해보자
    #0부정, 1긍정 → 폴더명 정렬해서 0,1,2 이런식으로 라벨링 neg-0 pos-1)

#시퀀스 생성
max_length = 600 #한평론에서 사용하는 단어의 최대길이
max_tokens = 20000 #고빈도사용단어 수

text_vectorization = TextVectorization(
    max_tokens = max_tokens,
    output_mode = "int", #텍스트를 정수 시퀀스로 변환
    output_sequence_length = max_length #단어임베딩층 사용하려면 반드시 모든 시퀀스길이 고정
)

text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds) #어휘사전 만들어야한다

int_train_ds = train_ds.map(lambda x, y:(text_vectorization(x), y), num_parallel_calls=1)
int_val_ds = val_ds.map(lambda x, y:(text_vectorization(x), y), num_parallel_calls=1)
int_test_ds = test_ds.map(lambda x, y:(text_vectorization(x), y), num_parallel_calls=1)

#임베딩 내부구조 엿보기
for item in int_train_ds:
    print(item)
    break

# 사전학습된 임베딩 데이터 불러오기
# https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt?select=glove.6B.100d.txt
#임베딩데이터, 단어별 각 단어와의 거리가 벡터로저장, 파일명의 100이 출력벡터크기
import numpy as np
# path_to_glove_file = "./Deep_Learning/data/glove.6B.100d.txt"
path_to_glove_file = "./data/glove.6B.100d.txt"
embedding_index = {}
with open(path_to_glove_file, encoding="utf-8") as f:
    for line in f: 
        word, coefs = line.split(maxsplit=1)
        #단어, 단어들간의 벡터구조로 돼있다. 예)the, 0.0012, 000172
        coefs = np.fromstring(coefs, "f", sep=" ") #나머지벡터들을 numpy배열로 전환
        embedding_index[word] = coefs
        # print(embedding_index) #궁금하니까 1개만 출력후 break
        # break
#{"the":[ , , , ,]}
print("개수", len(embedding_index))
#print(embeddings_index)

#우리데이터와 연동
vocabulary = text_vectorization.get_vocabulary() #우리 어휘사전 가져오기
#{단어:인덱스} 형태의 딕셔너리 생성
#{"", "[UNK]", "write", "love", "make", ...} vocabulary
#{0,1,2,3,4,5,6,...}
#zip ("",0) ("[UNK]",1) ("write", 2)....
#{"":0, "[UNK]":1, "write":2, ...}

word_index = dict(zip(vocabulary, range(len(vocabulary))))
embedding_dim = 100 #미리학습한 임베딩층 출력값100개
embedding_matrix = np.zeros((max_tokens, embedding_dim))
#케라스 Embedding레이어의 초기값
# embedding_matrix를 embedding_index 정보로 채워야한다.

#word_index는 {단어:인덱스}gudxodla
for word, i in word_index.items():
    #단어와 인덱스 가져온다
    if i < max_tokens: #혹시나 20000개를 넘어가는 토큰있을까봐 오류처리
        embedding_vector = embedding_index.get(word) #단어에 해당하는 벡터들 이동
    if embedding_vector is not None: #embedding_vector값이 None인 경우 제외
        embedding_matrix[i] = embedding_vector

print(embedding_matrix[:10])

from keras import models, layers
import tensorflow as tf

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(
    input_dim = max_tokens, 
    output_dim=embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    ####이부분필수. 사전학습된 층에 바꿔치기
    trainable=False, #임베딩가중치를 훈련중에 업데이트 할거냐? 사전학습된 인베딩층 사용할때 False
    mask_zero=True
    )(inputs)
print(embedded.shape)

#양방향 RNN
#          양방향감싸서 약방향처리     32개 유닛가진 LSTM층, 시퀀스데이터에서 장기의존성 포착하는데 사용
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

model.fit(int_train_ds, validation_data=int_val_ds, epochs=2, callbacks=callbacks)
print("테스트셋", model.evaluate(int_test_ds))
