# Ctrl + Shift + P → "Python: Select Interpreter" 입력해서 선택
# C:\ProgramData\Anaconda3\envs\deeplearning\python.exe
#conda activate deeplearning

from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(1234)

#1.데이터가져오기
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(type(train_images), type(train_labels))
#첫시작시 7만개 다운로드


#2.딥러닝 모델을 만든다.
from tensorflow.keras import models, layers

#네트워크 또는 모델이라고 부른다
#keras.Sequenctial 로 모델을 만드는데 매개변수로 list타입안에 레이어 객체를 전달한다

model = keras.Sequenctial([
#2-1.입력층 설계
#layel.Dense(출력값의개수, 활성화함수, 입력데이터의 크기-생략가능)
#출력값개수? 저 계층을 나왔을때 가져올 가중치들의 개수. 내마음대로 너무 크게 주면
#메모리 부족도 있고 , 과대적함 문제도 있음, 적당히, 2의 배수로 많이들 준다
    layers.Dense(64, activation='relu' ),
#2-2 중간에 다른층 추가 가능
#2-3 출력층,마지막층은 라벨에 맞추기.결과얻기위한층
#    손으로쓴 숫자이니 0~9 중에 하나여야한다.
#   딥러닝분류는 출력데이터를 확률로 반환
#   예)[0.1,0.1,0.05,0.7...] 결과는3으로 판단
#   각층 거치며 나오는값들은 실제확률이 아닌 엄청큰값들.
#   이를 모두합해 1이되는 확률로 전환해야하는데 이 함수가 softmax 함수
#   다중분류의 출력층의 활성화함수는 무조건ㄴ softmax함수
#내용 외워질때까지 연습해야함
    layers.Dense(10, activation='softmax') #출력값개수, 활성화함수


])