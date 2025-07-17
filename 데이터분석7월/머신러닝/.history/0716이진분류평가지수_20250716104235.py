#250716 AM9:20
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

# 1. California load_breast_cancer 데이터셋 로드

cancer = load_breast_cancer()
X = pd.DateFrame(cancer.data, columns= cancer.feature_names)
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

#5.선형회귀모델학습
model = LogisticRegression(random_state=1, solver='liblinear')
model.fit(X_train_scaled, y_train)