from konlpy.tag import Okt
okt = Okt()
print(okt.morphs("자연어 처리는 재밌습니다!"))

import tensorflow as tf
print("✅ TensorFlow version:", tf.__version__)
print("✅ GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 샘플 데이터
X = [[0], [1], [2], [3], [4]]
y = [[0], [1], [2], [3], [4]]

# 모델 정의
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=1)

pred = model.predict([[5]])
print("예측값:", pred)
