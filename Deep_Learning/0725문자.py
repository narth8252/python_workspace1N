#250725 PM1시
#영화평 데이터셋 imdb1.py
import keras
from keras.datasets import imdb
from keras import models
from keras import layers
import ternsorflow as tf
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
print(test_data.shape)
#데이터 자체도 궁금
print(train_data[:3]) #문장을 list타입으로 가져온다
print(train_labels[:3])

#데이터를 시퀀스로 바꿔야하는데 담주에 학습
#get_sord_index
word_index = imdb.get_word_index()
print(type(word_index))
print(word_index.keys())