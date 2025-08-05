#250731 PM3시 쌤PPT 딥러닝종합_백현숙.ppt 465p RNN 단어임베딩학습
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
#  ...that\'s about it.<br /><br />Three stars. Not a "Killer" comedy, but it tries.<br /><br />Rock on, Peace.'], shape=(3,), dtype=string)
# targets[0] tf.Tensor([0 1 0], shape=(3,), dtype=int32)
# (<tf.Tensor: shape=(32, 600), dtype=int64, numpy=
# array([[ 573,   38,    2, ...,    0,    0,    0],
#        [  15,   17,  105, ...,    0,    0,    0],
#        [  11,    7,   29, ...,    0,    0,    0],
#        ...,
#        [  11,   18,   14, ...,    0,    0,    0],
#        [ 109,    4,   20, ...,    0,    0,    0],
#        [   2, 1038,    5, ...,    0,    0,    0]], dtype=int64)>, <tf.Tensor: shape=(32,), dtype=int32, numpy=
# array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
#        1, 1, 1, 0, 0, 0, 0, 0, 0, 1])>)

# 4.모델구축: RNN과 단어임베딩 레이어
#예전엔 원핫인코딩으로 노가다했는데 지금은 임베딩으로 함
import tensorflow as tf
embedding_dim = 600
inputs = keras.Input(shape=(None,), dtype="int64") #모델 입력층
# embeded = tf.one_hot(inputs, depth=max_tokens) #에러남
# ... it makes you angry. Buy your kids the book instead."], shape=(3,), dtype=string)
# targets[0] tf.Tensor([0 1 0], shape=(3,), dtype=int32)
# (<tf.Tensor: shape=(32, 600), dtype=int64, numpy=
# array([[   4,   20, 1016, ...,    0,    0,    0],
#        [ 271,  717,    4, ...,    0,    0,    0],
#        [  11,  240,    5, ...,    0,    0,    0],
#        ...,
#        [  11,    3, 1552, ...,    0,    0,    0],
#        [  10,  103,   11, ...,    0,    0,    0],
#        [  10, 5198,   36, ...,    0,    0,    0]], dtype=int64)>, <tf.Tensor: shape=(32,), dtype=int32, numpy=
# array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1,
#        0, 1, 0, 0, 0, 0, 1, 1, 1, 1])>)

#위처럼하면 에러나서 아래로 바뀜: 
# Lambda 레이어를 사용한 One-Hot 인코딩 시도 (임베딩 레이어 대신)
from keras import models, layers
embedded = layers.Lambda(
    lambda x: tf.reshape(tf.one_hot(x, depth=max_tokens), (-1, tf.shape(x)[1], max_tokens)),  # Remove the extra dimension
    output_shape=(None, max_tokens)  # Specify the output shape
)(inputs)
#원핫인코딩 → 원핫인코딩 하고나서 모델에입력. 지금은 원핫인코딩 자체를 모델의 한단계로 추가.
#시퀀스 → 원핫인ㅋ딩 방법도 있고, 지금처럼 함수만들면 자동처리
print(embedded.shape) #Lambda레이어의 출력 형태 확인

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
