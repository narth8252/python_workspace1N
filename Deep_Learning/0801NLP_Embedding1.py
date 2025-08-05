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

from keras import models, layers
import tensorflow as tf
embedding_dim = 600
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim = max_tokens, output_dim=256)(inputs)
#입력벡터크기20000, 출력벡터크기256(의미없음 마음대로)
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

#250801
#임베딩층 - 내부적으로는 연산을 해서 단어와 단어사이의 관계를 계산해서 밀집벡터를 만든다.
#원핫인코딩 - 메모리를 너무 많이 차지함 최대한 한문장을 표현하는데 만일 최대 20000 단어까지 처리한다면
#한문장당 20000개가 필요 , 희소행렬 요소가 거의다 0인데 그중 몇개가 값이 있을때, 학습시 속도가 엄청 느리다
#임베딩층을 사용한다. => 단어와 단어사이의 거리를 재는 방식인데, 비슷 비슷한 단어가 같은 문장에 나타난다고 전제 하고 문제를 해결한다
#들릴수 있다.
#케라스가 Embedding 레이어를 제공한다. 이 레이어는 반드시 정수 인덱스를 받아야 한다.시퀀스받아서 밀집벡터 
# C:\ProgramData\anaconda3\envs\mytensorflow\Lib\site-packages\keras\layers\ __init__.py파일 안에 케라스있음. 

# (deeplearning) C:\Users\Admin\Documents\GitHub\python_workspace1N\Deep_Learning>python 0801NLP_Embedding1.py
# Found 20000 files belonging to 2 classes.
# 2025-08-01 09:32:09.503074: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2025-08-01 09:32:09.504934: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
# Found 5000 files belonging to 2 classes.
# Found 23215 files belonging to 2 classes.
# inputs.shape (32,)
# inputs.dtype <dtype: 'string'>
# targets.shape (32,)
# targets.dtype <dtype: 'int32'>
# inputs[0] tf.Tensor(
# [b'The opening sequence is supposed to show the Legion arriving in Paris on 13 Nov 1918. The troops pile off the train -- wearing the uniform in which the French Army, including the Legion, marched off to war in 1914! This a sure sign that the war flick you are about to see will be a turkey. (The French Army realized by 1915 that going to war in red trousers and dark blue overcoats was not working. Metropolitan French troops were put into "horizon blue" and Colonial troops were put into khaki.) The Claude Van-Damme (sp?) remake at least got the uniforms more or less right. Really is too bad when directors make these sorts of mistakes when they then go to all the effort to get other things right.'
#  b'This movie is excellent and I would recommend renting it for anyone whose local video store owns it. Or, even better, you could buy it because chances are you\'re going to watch this over and over. I can remember watching this movie as a kid and it was great back then. But after watching it again yesterday I\'ve found it to be amazing.<br /><br />A good blend of comedy (although not as great as "Mr Magoo"-another one of my favorites) and action. This deserves 10/10 and I\'m hoping that they will make a sequel soon (fingers crossed). If you do babysitting or have to look after young children for anything then I\'d recommend renting this movie as it will keep them entertained for hours :).'
#  b"The novel is easily superior and the best parts of the film are direct translations from what Greene wrote; for instance the quiet but grim humour that breaks into the scenes with Boyer and Lorre, or the murdered-child obsession that takes over some of the plot. Where the film deviates from the novel, it tends to the ludicrous.<br /><br />However I don't want to suggest that the film is bad in any way. It always looks the part and the story stays in the mind like a good 'un. Some of the minor characters were stock actors who could turn their hand to anything.<br /><br />It's a dreadful shame that the film's not available on DVD."], shape=(3,), dtype=string)
# targets[0] tf.Tensor([0 1 1], shape=(3,), dtype=int32)
# (<tf.Tensor: shape=(32, 600), dtype=int64, numpy=
# array([[   11,   120,     7, ...,     0,     0,     0],
#        [   11,    18,   626, ...,     0,     0,     0],
#        [   22,   241,    44, ...,     0,     0,     0],
#        ...,
#        [    2,   194,   429, ...,     0,     0,     0],
#        [10207, 10614,  3737, ...,     0,     0,     0],
#        [   11,     7,     2, ...,     0,     0,     0]], dtype=int64)>, <tf.Tensor: shape=(32,), dtype=int32, numpy=
# array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
#        1, 0, 0, 0, 0, 1, 0, 1, 1, 0])>)
# (None, None, 256)
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, None)]            0
#  embedding (Embedding)       (None, None, 256)         5120000
#  bidirectional (Bidirectiona  (None, 64)               73984
#  l)
#  dropout (Dropout)           (None, 64)                0
#  dense (Dense)               (None, 1)                 65
# =================================================================
# Total params: 5,194,049
# Trainable params: 5,194,049
# Non-trainable params: 0
# _________________________________________________________________
# Epoch 1/2
# 625/625 [==============================] - 582s 909ms/step - loss: 0.4807 - accuracy: 0.7847 - val_loss: 0.3628 - val_accuracy: 0.8542
# Epoch 2/2
# 625/625 [==============================] - 507s 810ms/step - loss: 0.3129 - accuracy: 0.8871 - val_loss: 0.3477 - val_accuracy: 0.8564
# 726/726 [==============================] - 194s 265ms/step - loss: 0.3590 - accuracy: 0.8518
# 테스트셋 [0.3589908182621002, 0.8518199324607849]

