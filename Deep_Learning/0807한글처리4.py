#250807 딥러닝복습_한글처리 코드복잡
#conda activate mytensorflow
#pip install konlpy (잘안되면 colab에 이 줄만 치면 끌어다 쓸수있음)
#pip install Korpora (네이버영화평)
#한국어 말뭉치 올라와있는 Kopora 사이트
# https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/nsmc.html
from multiprocessing import process
from operator import neg
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from konlpy.tag import Okt
import re
import os, pathlib, shutil, random
import numpy as np # Used for dummy dataset creation
from keras import layers

#네이버영화평데이터(NSMC)활용, 한국어텍스트 감성분석모델훈련 위한 전체 데이터 전처리 파이프라인을 매우 상세하게 구현
from Korpora import Korpora
Korpora.fetch("nsmc")
#저장위치: C:\Users\Admin(사용자계정)\Korpora\nsmc
corpus = Korpora.load("nsmc")
print(corpus. train[ : 3])

#데이터셋을 사용하기 위해서 해야할 작업
#네이버영화평 파일읽어서 폴더나누고 텍스트를 라벨보고 나눠넣기
def create_korean_dataset(base_dir="korean_imdb"):
    if os.path.exists(base_dir): #폴더가 기존재하면
        try:
            shutil.rmtree(base_dir)
        except OSError as e:
            print("error", e)

    #서브디렉토리 생성  train (훈련) → val (검증) → test (테스트)
    os.makedirs(os.path.join(base_dir, "train", "pos"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", "neg"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", "pos"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", "neg"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val", "pos"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val", "neg"), exist_ok=True)

    #파일은 train파일과 corpus가 이미 파일읽은상태임. train,test 쪽에 데이터 있음
    pos_train_texts = []
    neg_train_texts = []
    for i in range(len(corpus.train)):
        if corpus.train[i].label==1:
            pos_train_texts.append(corpus.train[i].text)
        else:
            neg_train_texts.append(corpus.train[i].text)

    pos_test_texts = []
    neg_test_texts = []
    for i in range(len(corpus.test)):
        if corpus.test[i].label==1:
            pos_test_texts.append(corpus.test[i].text)
        else:
            neg_test_texts.append(corpus.test[i].text)

    #훈련셋과 검증셋 나누기
    pos_train_texts = pos_train_texts[:1000]
    neg_train_texts = pos_train_texts[:1000]
    pos_val_texts = pos_train_texts[:1000]
    neg_val_texts = pos_train_texts[:1000]

    for i, text in enumerate(pos_train_texts):
        with open(os.path.join(base_dir, "train", "pos", f"pos_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    for i, text in enumerate(neg_train_texts):
        with open(os.path.join(base_dir, "train", "neg", f"neg{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    for i, text in enumerate(pos_val_texts):
        with open(os.path.join(base_dir, "val", "pos", f"pos_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    for i, text in enumerate(neg_val_texts):
        with open(os.path.join(base_dir, "val", "neg", f"neg{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    for i, text in enumerate(pos_test_texts):
        with open(os.path.join(base_dir, "test", "pos", f"pos_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    for i, text in enumerate(neg_test_texts):
        with open(os.path.join(base_dir, "test", "neg", f"neg{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)


    print("작업완료")
    return base_dir

# 1. 데이터셋 생성
# korean_data_dir = create_korean_dataset("./data/naver_imdb") #작업완료후 주석처리하고 아래껄로 사용
korean_data_dir = "./data/naver_imdb"

#tensorflow가 numpy를 안쓰고 vectorization을 쓰고있기때문에 그쪽으로 맞추자
# 2. 데이터셋 로드 및 초기화
#keras.utils.text_dataset_from_directory 함수사용(데이터읽어올 '준비'하기)
batch_size = 32
train_ds_raw = keras.utils.text_dataset_from_directory(
    korean_data_dir + "/train", batch_size=batch_size, label_mode="binary")
val_ds_raw = keras.utils.text_dataset_from_directory(
    korean_data_dir + "/val", batch_size=batch_size, label_mode="binary")
test_ds_raw = keras.utils.text_dataset_from_directory(
    korean_data_dir + "/test", batch_size=batch_size, label_mode="binary")

######## 한글이나 비영어권 국가들 ########
# 3. 한국어 텍스트 전처리 함수 준비
okt = Okt()

######## def clean_text() 함수가 호출을 직접해서 tensor에 넘겨주는게 아님
#train_ds_raw를 이용해서 파일을 읽는건 keras가 읽어서 우리한테 뭐로 주느냐
#Tensorflow 용어의 시작은 Tensor(벡터) → Tensor(벡터)
#데이터가 텐서들을 타고 흐른다고 해서 Tensorflow
#Tensorflow의 tensor로 준다. 문자셋이 인코딩후 전달
#\U05\u0x  16진수로 변경해서 온다
##⭐디코딩작업 필수
def clean_text(text):
    text = text.decode("utf-8") #⭐tf.tensor → python의 str타입으로 변경
    text = text.lower() #대문자 → 소문자
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
    return text

#tf.tensor형태로 데이터를 받아서 → str으로 바꿔서 정규화(불필요한 문자삭제),토큰나눠서 
#tf.tensor로 바꿔야한다 (매개변수도 텐서플로우.텐서, 리턴도 텐서플로우.텐서)
def python_korean_preprocess(text_tensor): #매개변수는 tf.tensor타입
    processed_text = [] #여기에 처리한 문자열저장
    for text_bytes in text_tensor.numpy():
        #인코딩된 데이터를 전달받음
        cleaned_text = clean_text(text_bytes) 
        #tf.tensor → 정제후 토큰쪼개기 → string타입으로 받는다
        morphed_text = " ".join(okt.morphs(cleaned_text))
        processed_text.append(morphed_text)
        #list → string ==> Tensor로 전환해서 보내야한다

    return tf.constant(processed_text, dtype=tf.string)

#tf.py_funtion 함수: python함수를 tesorflow에 끼워넣기
def tf_korean_preprocess_fn(texts, labels):
    processed_texts = tf.py_function(
        func = python_korean_preprocess, #전달함수
        inp = [texts], #입력데이터
        Tout =tf.string #출력타입
    )
    #명시적으로 타입 지정
    processed_texts.set_shape(texts.get_shape())
    return processed_texts, labels

#TextVectorization 생성(어휘사전)
max_tokens = 10000 #어휘사전은 자주쓰는 단어 10000개만
output_sequence = 20
vectorizer = TextVectorization(
    max_tokens = max_tokens,
    output_mode = "int",
    output_sequence_length = output_sequence,
    standardize = None,
    split = "whitespace"  # 공백 기준으로 토큰 분리
)


#모든데이터셋에대해 tf_korean_preprocess_fn 함수처리 해야함
#map함수는 연산수행해 변환
#texts, labels 각 요소를 하나씩 전달후 연산수행후 반환
#num_parallel_calls = tf.data.AUTOTUNE : 시스템상태에 따른 적당한 병행처리(직접 개수지정가능)
train_ds_processed = train_ds_raw.map(tf_korean_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)    
val_ds_processed = val_ds_raw.map(tf_korean_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)    
test_ds_processed = test_ds_raw.map(tf_korean_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)    
print("--- 전처리 완료 ---")

#어휘사전 생성
vectorizer.adapt(train_ds_processed.map(lambda x, y : x)) #x:text, y:label

#실제학습하려면 내 데이터를 벡터화 시켜야함
def vectorize_text_fn(texts, labels):
    return vectorizer(texts),labels #벡터화시켜 반환

def study():
    train_ds_vectorized = train_ds_processed.map(vectorize_text_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_vectorized = val_ds_processed.map(vectorize_text_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds_vectorized = test_ds_processed.map(vectorize_text_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # 데이터 전처리 파이프라인의 마지막 단계로, 모델 학습의 속도와 효율성을 최적화하는 역할
    # 전처리된 데이터셋에 벡터화 함수를 적용하여 학습에 바로 사용할 수 있는 최종 데이터셋을 만드는 과정
    #데이터셋이 사용하게 CPU임 캐쉬랑 프리패치
    train_ds_vectorized = train_ds_vectorized.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds_vectorized = val_ds_vectorized.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds_vectorized = test_ds_vectorized.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    #결과확인
    print(vectorizer.get_vocabulary()[:20])
    for text_batch, label_batch in train_ds_vectorized.take(1): #1사이클만 가져와서
        print(text_batch.shape)
        print(label_batch.shape)
        print(text_batch[0, :10].numpy())

        #역변환하여 확인
        vocabulary = vectorizer.get_vocabulary()
        decoded = " ".join(vocabulary[idx] for idx in text_batch[0, :10].numpy() if idx>1)
        print(decoded)

    print("--- 최종 데이터셋 준비작업 완료 ---")

    ################# 딥러닝: Embedding(밀집벡터) → 단어 사이의 거리 표현 ##########################3
    #################        원핫인코딩 : 희소행렬, 차원이 너무 거대
    voca_size = vectorizer.vocabulary_size()
    embedding_dim = 128 #대충씀. 특별히 사전에 학습된 내용을 사용하는게 아니면 차원은 내맘대로
    inputs = keras.Input(shape=(None,), dtype=tf.int64)
    x = layers.Embedding(
        input_dim=voca_size,
        output_dim=embedding_dim,
        mask_zero=True
    )(inputs)
    x = layers.Bidirectional(layers.LSTM(32))(x) #순환신경망
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x) #이진분류라서
    model = keras.Model(inputs, outputs)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint("korean_rnn_sentiment_model.keras",
                                        save_best_only=True,
                                        monitor='val_accuracy',
                                        mode = 'max')
    ]

    import pickle
    print("\n --- 모델 훈련 시작 ---")
    history = model.fit(train_ds_vectorized,
                        validation_data=val_ds_vectorized,
                        epochs=5,
                        callbacks=callbacks)
    with open("korean_rnn_sentiment_history", "wb") as f:
        pickle.dump(history.history, f)
    print("--- korean_rnn_sentiment_히스토리 저장 완료 ---")

import pickle
def predict():
    model = keras.models.load_model("korean_rnn_sentiment_model.keras")
    with open("korean_rnn_sentiment_history", "rb") as f:
        history = pickle.load(f)

    sample = [
        "이 영화는 정말 감동적이었어요.",    # ===> 긍정 0.5010543
        "영화수준이 너무 낮아요.",           # ===> 긍정 0.5017421
        "반드시 봐야할 영화입니다.",         # ===> 부정 0.4992522
        "배우 아니면 이 영화를 안봤을 거예요." # ===> 부정 0.49979416
    ]

    #샘플도 클린하게
    processed_sample = [
        " ".join(okt.morphs(clean_text(text.encode('utf-8')))) for text in sample
    ]

    vectorize_sample = vectorizer(tf.constant(processed_sample, dtype=tf.string))
    prediction = model.predict(vectorize_sample)
    #확률
    prob = prediction.flatten()
    for i, text in enumerate(sample):
        sentiment = "긍정" if prob[i]>=0.5 else "부정"
        print(text[:40], "===>", sentiment, prob[i])
        
# study()  
predict()

# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Deep_Learning>python 0807한글처리4.py
# 2025-08-07 14:50:12.033265: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-08-07 14:50:18.291193: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_train.txt
# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_test.txt

#     Korpora 는 다른 분들이 연구 목적으로 공유해주신 말뭉치들을
#     손쉽게 다운로드, 사용할 수 있는 기능만을 제공합니다.

#     말뭉치들을 공유해 주신 분들에게 감사드리며, 각 말뭉치 별 설명과 라이센스를 공유 드립니다.
#     해당 말뭉치에 대해 자세히 알고 싶으신 분은 아래의 description 을 참고,
#     해당 말뭉치를 연구/상용의 목적으로 이용하실 때에는 아래의 라이센스를 참고해 주시기 바랍니다.

#     # Description
#     Author : e9t@github
#     Repository : https://github.com/e9t/nsmc
#     References : www.lucypark.kr/docs/2015-pyconkr/#39

#     Naver sentiment movie corpus v1.0
#     This is a movie review dataset in the Korean language.
#     Reviews were scraped from Naver Movies.

#     The dataset construction is based on the method noted in
#     [Large movie review dataset][^1] from Maas et al., 2011.

#     [^1]: http://ai.stanford.edu/~amaas/data/sentiment/

#     # License
#     CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
#     Details in https://creativecommons.org/publicdomain/zero/1.0/

# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_train.txt
# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_test.txt
# LabeledSentence(text=('아 더빙.. 진짜 짜증나네요 목소리', '흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나', '너무재밓었다그래서보는것을추천한다'), label=[0, 1, 0])
# Found 2000 files belonging to 2 classes.
# 2025-08-07 14:50:38.571465: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 2000 files belonging to 2 classes.
# Found 50000 files belonging to 2 classes.
# --- 전처리 완료 ---
# 2025-08-07 14:51:17.553294: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
# ['', '[UNK]', '이', '영화', '의', '에', '가', '을', '를', '들', '도', '는', '정말', '은', '한', '최고', '적', '너무', '다', '잘']
# (32, 20)
# (32, 1)
# [ 189  193   43   41    9   10 1183 1261  191  141]
# 2025-08-07 14:51:20.849339: W tensorflow/core/kernels/data/cache_dataset_ops.cc:914] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
# 한국 전쟁 때 그 들 도 르완다 난민 처럼 우리
# 2025-08-07 14:51:21.173903: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
# --- 최종 데이터셋 준비작업 완료 ---
# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ input_layer (InputLayer)      │ (None, None)              │               0 │ -                          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ embedding (Embedding)         │ (None, None, 128)         │         566,784 │ input_layer[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ not_equal (NotEqual)          │ (None, None)              │               0 │ input_layer[0][0]          │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ bidirectional (Bidirectional) │ (None, 64)                │          41,216 │ embedding[0][0],           │
# │                               │                           │                 │ not_equal[0][0]            │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dropout (Dropout)             │ (None, 64)                │               0 │ bidirectional[0][0]        │
# ├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
# │ dense (Dense)                 │ (None, 1)                 │              65 │ dropout[0][0]              │
# └───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
#  Total params: 608,065 (2.32 MB)
#  Trainable params: 608,065 (2.32 MB)
#  Non-trainable params: 0 (0.00 B)

#  --- 모델 훈련 시작 ---
# Epoch 1/5
# 2025-08-07 14:51:33.009969: E tensorflow/core/util/util.cc:131] oneDNN supports DT_BOOL only on platforms with AVX-512. Falling back to the default Eigen-based implementation if present.
# 63/63 ━━━━━━━━━━━━━━━━━━━━ 32s 301ms/step - accuracy: 0.5178 - loss: 0.6942 - val_accuracy: 0.5000 - val_loss: 0.6932
# Epoch 2/5
# 63/63 ━━━━━━━━━━━━━━━━━━━━ 2s 28ms/step - accuracy: 0.5027 - loss: 0.6937 - val_accuracy: 0.5000 - val_loss: 0.6932
# Epoch 3/5
# 63/63 ━━━━━━━━━━━━━━━━━━━━ 2s 29ms/step - accuracy: 0.4672 - loss: 0.6942 - val_accuracy: 0.5000 - val_loss: 0.6932
# Epoch 4/5
# 63/63 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - accuracy: 0.5018 - loss: 0.6933 - val_accuracy: 0.5000 - val_loss: 0.6932
# Epoch 5/5
# 63/63 ━━━━━━━━━━━━━━━━━━━━ 2s 30ms/step - accuracy: 0.4965 - loss: 0.6936 - val_accuracy: 0.5000 - val_loss: 0.6932
# --- korean_rnn_sentiment_히스토리 저장 완료 ---

# (mytensorflow) C:\Users\Admin\Documents\GitHub\python_workspace1N\Deep_Learning>python 0807한글처리4.py
# 2025-08-07 16:03:04.278660: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-08-07 16:03:12.348998: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_train.txt
# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_test.txt

#     Korpora 는 다른 분들이 연구 목적으로 공유해주신 말뭉치들을
#     손쉽게 다운로드, 사용할 수 있는 기능만을 제공합니다.

#     말뭉치들을 공유해 주신 분들에게 감사드리며, 각 말뭉치 별 설명과 라이센스를 공유 드립니다.
#     해당 말뭉치에 대해 자세히 알고 싶으신 분은 아래의 description 을 참고,
#     해당 말뭉치를 연구/상용의 목적으로 이용하실 때에는 아래의 라이센스를 참고해 주시기 바랍니다.

#     # Description
#     Author : e9t@github
#     Repository : https://github.com/e9t/nsmc
#     References : www.lucypark.kr/docs/2015-pyconkr/#39

#     Naver sentiment movie corpus v1.0
#     This is a movie review dataset in the Korean language.
#     Reviews were scraped from Naver Movies.

#     The dataset construction is based on the method noted in
#     [Large movie review dataset][^1] from Maas et al., 2011.

#     [^1]: http://ai.stanford.edu/~amaas/data/sentiment/

#     # License
#     CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
#     Details in https://creativecommons.org/publicdomain/zero/1.0/

# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_train.txt
# [Korpora] Corpus `nsmc` is already installed at C:\Users\Admin\Korpora\nsmc\ratings_test.txt
# LabeledSentence(text=('아 더빙.. 진짜 짜증나네요 목소리', '흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나', '너무재밓었다그래서보는것을추천한 다'), label=[0, 1, 0])
# Found 2000 files belonging to 2 classes.
# 2025-08-07 16:03:33.700155: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Found 2000 files belonging to 2 classes.
# Found 50000 files belonging to 2 classes.
# --- 전처리 완료 ---
# 2025-08-07 16:04:25.507091: I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
# 2025-08-07 16:04:27.859093: E tensorflow/core/util/util.cc:131] oneDNN supports DT_BOOL only on platforms with AVX-512. Falling back to the default Eigen-based implementation if present.
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
# 이 영화는 정말 감동적이었어요. ===> 긍정 0.5010543
# 영화수준이 너무 낮아요. ===> 긍정 0.5017421
# 반드시 봐야할 영화입니다. ===> 부정 0.4992522
# 배우 아니면 이 영화를 안봤을 거예요. ===> 부정 0.49979416