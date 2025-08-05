#250731 AM10시 쌤PPT 딥러닝종합_백현숙.ppt 433p
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
    break #하나만 출력해보자 
    #0부정, 1긍정 → 폴더명 정렬해서 0,1,2 이런식으로 라벨링 neg-0 pos-1

#데이터셋이 문장과 라벨로 연력되어, 라벨은 이미 정수화돼서 오니까 버리고, 문장만 벡터화
text_vectorization = TextVectorization(
    max_tokens = 20000, #자주사용단어20000개 지정
    output_mode = "count", #빈도수
    # output_mode = "tf_idf", #
    # output_mode = "multi_hot", #벡터로 각리뷰마다 20000개의 요소갖는 배열생성
    #"multi_hot"인코딩 또는 BoW(Bag of Word)" "배열에서 문장중에 단어있느곳1, 없으면0
    ngrams=2

)

#가져온 train_ds에서 문장만필요
text_only_train_ds = train_ds.map(lambda x, y:x) #데이터셋으로부터 문장만 추출
text_vectorization.adapt(text_only_train_ds) #어휘사전 만들기

###데이터별로 이 작업 진행
#유니그램 - 단어를 1개씩 가져오는 방식
#멀티프로세싱, 한번에 CPU코어4개사용해서 작업수행
binary_1gram_train_ds = train_ds.map(lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_val_ds = val_ds.map(    lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_test_ds = test_ds.map(  lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)

print("----- 벡터화 후 -----")
for inputs, targets in binary_1gram_train_ds: #실제 읽어오는 데이터 확인
    print("inputs.shape", inputs. shape)
    print("inputs.dtype", inputs.dtype)
    print("targets.shape", targets.shape)
    print("targets.dtype", targets.dtype)
    print("inputs[0]", inputs[:3])
    print("targets[0]", targets[:3])
    break #하나만 출력해보자

#모델 만들어서 반환하는 함수
from keras import layers, models
def getModel(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,)) #입력층 만들기
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

#     return model

model = getModel()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_2gram.keras", save_best_only=True)
]
#적절한 시점에 파일저장
#cache : 데이터셋을 메모리에 캐싱. 첫에포크에서 전처리1번만 하고 더이상안하고 재사용
#메모리에 들어갈만큼 작은 데이터셋일때만 가능
model.fit(binary_1gram_train_ds.cache(),
          validation_data = binary_1gram_val_ds,
          epochs=10,
          callbacks=callbacks)

model = models.load_model("binary_2gram.keras") #학습한 내용읽어보기
print("테스트셋 정확도", model.evaluate(binary_1gram_test_ds)) #손실도, 정확도


#예측하기
inpits = keras.Input(shape=(1,), dtype="string") #1문장 넣기
processed_inputs = text_vectorization(inputs) #벡터화
outputs = model(processed_inputs)
inference_model = keras.Model(inputs, outputs) #모델만 생성

raw_text_data = tf.convert_to_tensor([
    ["That was an excellent movie I love it"]
])

predictions = inference_model(raw_text_data) #반환값은 1이 될 확률(부정0, 긍정1)
print("긍정적일 확률: ", predictions[0]*100)

# ...  Other than this I feel sure that this is a film you will really enjoy."], shape=(3,), dtype=string)      
# targets[0] tf.Tensor([1 1 1], shape=(3,), dtype=int32)
# ----- 벡터화 후 output_mode = "count"-----
# inputs.shape (32, 20000)
# inputs.dtype <dtype: 'float32'>
# targets.shape (32,)
# targets.dtype <dtype: 'int32'>
# inputs[0] tf.Tensor(
# [[140.   9.   6. ...   0.   0.   0.]
#  [101.  11.   6. ...   0.   0.   0.]
#  [ 57.   7.   4. ...   0.   0.   0.]], shape=(3, 20000), dtype=float32)
# targets[0] tf.Tensor([0 0 1], shape=(3,), dtype=int32)
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 20000)]           0
#  dense (Dense)               (None, 16)                320016
#  dropout (Dropout)           (None, 16)                0
#  dense_1 (Dense)             (None, 1)                 17
# =================================================================
# Total params: 320,033
# Trainable params: 320,033
# Non-trainable params: 0
# _________________________________________________________________
# Epoch 1/10
# 625/625 [==============================] - 68s 106ms/step - loss: 0.4733 - accuracy: 0.7864 - val_loss: 0.3248 - val_accuracy: 0.8826
# ...
# Epoch 10/10
# 625/625 [==============================] - 12s 19ms/step - loss: 0.1991 - accuracy: 0.9269 - val_loss: 0.3730 - val_accuracy: 0.8782
# 726/726 [==============================] - 109s 148ms/step - loss: 0.2828 - accuracy: 0.8925
# 테스트셋 정확도 [0.28281137347221375, 0.892526388168335]