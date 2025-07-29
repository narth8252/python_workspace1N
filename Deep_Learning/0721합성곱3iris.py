#250721 PM2시
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# 데이터 불러오기
data = load_iris()
X = data.data     # (150, 4)
y = data.target   # (150,)

# 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 원-핫 인코딩
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# 2. 딥러닝 모델 생성 및 학습
from tensorflow.keras import models, layers

def make_iris_model():
    model = models.Sequential([
        layers.Dense(16, activation='relu', input_shape=(4,)),  # 특성 4개
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')                   # 클래스 3개
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = make_iris_model()

model.fit(X_train, y_train_cat, epochs=50, batch_size=8, validation_split=0.1)

# 3. 평가 및 예측
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"테스트셋 손실: {loss:.4f}, 정확도: {acc:.4f}")


# Epoch 1/50
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 3s 35ms/step - accuracy: 0.5162 - loss: 1.0202 - val_accuracy: 0.8333 - val_loss: 0.8784
# Epoch 2/50
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.6723 - loss: 0.9512 - val_accuracy: 0.7500 - val_loss: 0.8176
# Epoch 3/50
# ...
# Epoch 47/50
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.9359 - loss: 0.1392 - val_accuracy: 1.0000 - val_loss: 0.1073
# Epoch 48/50
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9887 - loss: 0.0784 - val_accuracy: 1.0000 - val_loss: 0.0978
# Epoch 49/50
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9477 - loss: 0.1220 - val_accuracy: 1.0000 - val_loss: 0.0973
# Epoch 50/50
# 14/14 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.9519 - loss: 0.1082 - val_accuracy: 1.0000 - val_loss: 0.0904
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - accuracy: 0.9000 - loss: 0.1701
# 테스트셋 손실: 0.1701, 정확도: 0.9000

#▼ris데이터셋 딥러닝 학습 결과 분석
# 훈련 및 검증 정확도/손실 추이
# • 초기 에포크(1~10):
# ㆍ훈련 정확도는 0.52 → 0.86까지 증가
# ㆍ검증 정확도는 에포크마다 0.66~0.83 사이에서 출발하여 안정화됨
# ㆍ손실은 점차 감소하는 전형적인 양상
# • 중반 이후(11~30):
# ㆍ훈련 정확도는 꾸준히 0.90 이상 유지
# ㆍ검증 정확도는 0.75~0.92, 이후 40에포크 즈음부터 1.00(100%) 도달
# ㆍ검증 손실 역시 지속적으로 하락
# • 마지막(40~50):
# ㆍ검증 정확도는 1.00(100%)을 꾸준히 기록
# ㆍ훈련 정확도도 거의 0.95 이상, 손실도 매우 낮음

# • 최종 테스트셋 결과
# ㆍ테스트셋 손실: 0.1701
# ㆍ테스트셋 정확도: 0.9000 (90%)

# • 종합 평가
# ㆍ50에포크 훈련 과정에서, 모델의 학습 성능과 검증 성능 모두 안정적으로 상승해 오버피팅 없이 좋은 일반화 능력을 보여줌
# ㆍ테스트셋에서 90%의 정확도를 달성해, iris 분류 문제에 딥러닝 모델을 직접 적용해도 충분히 높은 성과를 얻을 수 있음을 확인

# • 학습 성과 및 참고사항
# ㆍ입력 데이터의 차원 변환(reshape)이 필요 없는 벡터·표 형태 데이터라는 특징이 확실히 반영되었음
# ㆍ모델 구성(예: Dense, softmax)은 간단하지만 작은 데이터셋에서도 좋은 분류 성능을 낼 수 있음
# ㆍepoch에 따라 손실 감소와 정확도 상승이 모두 꾸준히 관찰됨
# • 구분	          값
# 최종 훈련셋 정확도	약 95% 이상
# 최종 검증셋 정확도	100%
# 최종 테스트셋 정확도	90%
# 테스트셋 손실	       0.17
# ㆍ이 실험 결과는 iris와 같은 전형적인 벡터 데이터에 대해 딥러닝도 매우 효과적임을 보여줍니다.
# ㆍ표준 스케일링, 원-핫 인코딩 후 별도 차원변환 없이 바로 사용할 수 있다는 점도 잘 확인