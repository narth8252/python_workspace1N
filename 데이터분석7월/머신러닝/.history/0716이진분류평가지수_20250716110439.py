#250716 AM10시
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic' #내컴파이썬에 설치된 폰트명으로 변경
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # 특성 스케일링 (선형 모델에 중요)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer # 유방암 데이터셋

#1.load_breast_cancer 데이터셋 로드
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns= cancer.feature_names)
y = cancer.target
y_changed = np.where(y==0, 1, 0)

print("---체인지전---")
print(y[:20])
print("---체인지후---")
print(y_changed[:20])

#2.불균형셋 데이터 여부확인
print(dict(zip(*np.unique(y_changed, return_counts=True))))

#3. 데이터셋 분할 (훈련, 테스트) - numpy 2차원(dataFrame),1차원(series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4.특성스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #비지도학습 fit -> transtorm
X_test_scaled = scaler.fit_transform(X_test) #비지도학습 fit -> transtorm

#5.모델학습
model = LogisticRegression(random_state=1, solver='liblinear')
model.fit(X_train_scaled, y_train)

#6.테스트데이터로 예측
y_pred = model.predict(X_test_scaled)
#양성예측한것모두, 악성예측확률1개만 가져오기
y_pred_proba = model.predict_proba(X_test_scaled)[:,1]

#7.오차행렬 구하기
cm = confusion_matrix(y_test, y_pred)
print(cm)
# ---체인지전---
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
# ---체인지후---
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0]
# {np.int64(0): np.int64(357), np.int64(1): np.int64(212)}
# [[41  2]
#  [ 0 71]]

#8.분류리포트
re