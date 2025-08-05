#250731 AM10시 쌤PPT 딥러닝종합_백현숙.ppt 433p
#자연어처리NLP 정규식
import re
import string 
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization

def custom_standardization_fn(text):
    # string.punctuation 사용 (정확한 철자)
    lower_text = tf.strings.lower(text)
    return tf.strings.regex_replace(lower_text, f"[{re.escape(string.punctuation)}]", "")

def custom_split_fn(text):
    return tf.strings.split(text)

#객체 만들때 파라미터 값 지정
text_vectorization = TextVectorization(
    output_mode = "int",                    #출력이 시퀀스임
    standardize= custom_standardization_fn, #표준화에 사용할 함수 전달
    split= custom_split_fn                  #토큰화에 사용할 함수 전달
)

# 데이터셋 준비
dataset=[
    "I write, erase, reqrite",
    "Erase again, and then",
    "A poppy blooms",
    "Dog is qute"
]

text_vectorization.adapt(dataset) #학습시킬 데이터가 있으면 adapt에 전달

#단어 빈도수로 자동정리
vocabulary = text_vectorization.get_vocabulary()
print(vocabulary)

#인코딩
# def encode(self, text):
text = "I write, rewrite, and still rewrite again"
encoded = text_vectorization(text)
print(encoded)

#디코딩
decoded_voca = dict(enumerate(vocabulary))
print(decoded_voca)

decoded_sen = " ".join(decoded_voca[int(i)] for i in encoded)
print(decoded_sen)