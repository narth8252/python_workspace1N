import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical #딥러닝은 라벨도 원핫인코딩 필요
tf.random.set_seed(1)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기
iris = load_iris()
X = iris.data              # 4개의 특성
y = iris.target            # 3개의 클래스: 0, 1, 2

# 2. 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 5. 네트워크 구축
network = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),   # 입력층 + 히든1
    keras.layers.Dense(8, activation='relu'),                      # 히든2 - 추가 가능
    keras.layers.Dense(3, activation='softmax')                    # 출력층: 클래스 3개
])

# 6. 컴파일
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 학습
history = network.fit(X_train, y_train, epochs=3, batch_size=100)

# 8. 평가
test_loss, test_acc = network.evaluate(X_test, y_test)
print(f'\n 테스트 정확도: {test_acc:.4f}, 손실: {test_loss:.4f}')
# Epoch 1/3
# 2/2 [==============================] - 2s 10ms/step - loss: 1.2175 - accuracy: 0.2000
# Epoch 2/3
# 2/2 [==============================] - 0s 6ms/step - loss: 1.1839 - accuracy: 0.2083
# Epoch 3/3
# 2/2 [==============================] - 0s 7ms/step - loss: 1.1650 - accuracy: 0.2167
# 1/1 [==============================] - 1s 576ms/step - loss: 1.1416 - accuracy: 0.1667
#  테스트 정확도: 0.1667, 손실: 1.1416

# 모델 구성 요소 설명
# StandardScaler: 특성값들을 정규화해서 학습 성능 향상
# to_categorical: 다중 클래스니까 one-hot 인코딩 필요
# Dense: 완전연결층, 활성화 함수는 relu와 softmax
# categorical_crossentropy: 다중 클래스 분류에 꼭 필요한 손실함수
# batch_size=100: 데이터가 150개뿐이라 100은 거의 전체 배치
# epochs=3: 반복횟수 적게 줘서 빠르게 테스트 가능