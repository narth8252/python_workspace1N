import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

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
print(f'\n📊 테스트 정확도: {test_acc:.4f}, 손실: {test_loss:.4f}')
