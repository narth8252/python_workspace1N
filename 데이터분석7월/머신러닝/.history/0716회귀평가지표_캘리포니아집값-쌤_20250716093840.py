#250716 AM9:20
import pandas as pd
import numpy as np
import optuna
from sklearn.datasets import fetch_california_housing #"회귀Regression문제" 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 사이킷런(Scikit-learn)의 fetch_california_housing 데이터셋은 
# 1990년 캘리포니아 인구 조사 데이터를 기반으로 하는 주택 가격 예측 회귀 모델 학습용 데이터셋입니다. 
# 머신러닝, 특히 회귀 분석을 처음 접하는 입문자에게 적합한 데이터로 널리 사용됩니다.
# 이 데이터셋은 캘리포니아의 각 블록 그룹(최소 지리적 단위)에 대한 정보를 담고 있으며, 
# 총 20,640개의 샘플과 8개의 특성(feature)으로 구성되어 있습니다.

# 8개의 주요특성 (Features)
# MedInc              블록 그룹의 중간 소득
# HouseAge            블록 그룹 내 주택의 중간 연령
# AveRooms            가구당 평균 방 수
# AveBedrms            가구당 평균 침실 수
# Population           블록 그룹의 인구
# AveOccup              가구당 평균 구성원 수
# Latitude              위도
# Longitude             경도

# 타겟 변수 (Target Variable)
# 이 데이터셋의 목표(target)는 해당 블록 그룹의 중간 주택 가격(MedHouseVal)을 예측하는 것입니다. 
# 가격은 10만 달러 단위로 표시됩니다.

# 데이터셋의 특징 및 활용
#  • 실제 데이터 기반: 1990년 미국 인구 조사라는 실제 데이터를 기반으로 하고 있어 현실적인 문제 해결 능력을 기르는 데 도움이 됩니다.
#  • 회귀 문제에 적합: 연속적인 값(주택 가격)을 예측하는 회귀(Regression) 모델을 학습하고 평가하는 데 이상적입니다.
#  • 전처리 필요성: 데이터에 따라 특성들의 스케일이 다르므로, 모델의 성능을 높이기 위해 데이터 스케일링과 같은 전처리 과정의 필요성을 학습할 수 있습니다. 예를 들어, 소득(MedInc)과 인구(Population)는 값의 범위가 크게 다릅니다.
#  • 시각화: 위도와 경도 데이터를 활용하여 캘리포니아 지도 위에 주택 가격 분포를 시각화하는 등 데이터 탐색 및 분석 연습에 유용합니다.

# 1. California Housing 데이터셋 로드
print("----- California Housing 데이터셋 로드 중 -----")
housing = fetch_california_housing()
print(type(housing)) #딥러닝(Tensorflow:C++) → 케라스(초보자용),파이토치(세밀제어가능)

#차트그리려면 numpy → 요소하나씩 그려야해서 차트그리고싶다.
#DataFrame: 데이타프레임자체가 차트제공하기도 하고, seaborn, plotly(interactive)차트 있다.
#python코드로 차트그리면 플로틀리는 html과 CSS와 자바스크립트로 움직이는 차트가 만들어진다.
#예전에는 R언어만 지원했는데 현재는 파이썬 라이브러리가 지원(예은이찾음)
#넘파이배열, 컬럼명
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="houseval")

print(X.head())
print(X.columns)
print(y[:10])

#2. 데이터셋 분할 (훈련, 테스트) - numpy 2차원(dataFrame),1차원(series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3.결측치,이상치없음
#4.스케일링 또는 표준화, 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #비지도학습 fit -> transtorm
X_test_scaled = scaler.fit_transform(X_test) #비지도학습 fit -> transtorm

#선혀