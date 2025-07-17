from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(1234)

#1.데이터가져오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(type(train_images), type(train_labels))
#첫시작시 7만개 다운로드
# Ctrl + Shift + P → "Python: Select Interpreter" 입력해서 선택
# C:\ProgramData\Anaconda3\envs\deeplearning\python.exe
#conda activate deeplearning
#conda activate deeplearning

#2.딥러닝 모델을 만든다.
from tensorflow.keras import models, layers

#네트워크 또는 모델이라고 부른다
#keras.Sequenctial 로 모델을 만드는데 매개변수로 list타입안에 레이어 객체를 전달한다

model = keras.Sequenctial([
#2-1.7\입력층을 설계한다
#layel.Dense(입력값의개수, 활성화함수, 입력데이터의 크기-생략가능)
#출력값의 개수? 저 계층을 나왔을때가 가져올 가중치들의 개수 내마음대로 너무 크게 주면
#메모리 부족도 있고 , 과대적함 문제도 있음, 적당히, 2의 배수로 많이들 준다
layers.Dense(64, activation='relu' )
#2-2 중간에 다른층 추가 가능


])