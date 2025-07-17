#250716 AM9:20
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing # 캘리포니아 집값 데이터셋
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # 특성 스케일링 (선형 모델에 중요)

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
# 예측에 사용할 8개의독립변수(특성)들을 보기편한 표 데이터프레임으로 만듭니다. columns에 특성이름지정하여 각열의 의미 명확히
y = pd.Series(housing.target, name="houseval")
# 예측 목표값인 종속변수(주택가격)를 시리즈형태로 만듭니다.

print(X.head())
print(X.columns)
print(y[:10])

#2. 데이터셋 분할 (훈련, 테스트) - numpy 2차원(dataFrame),1차원(series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test_size=0.2: 전체데이터중 **20%**를 테스트용으로, 나머지 **80%**를 훈련용으로 사용합니다.
# random_state=42: 코드여러번실행해도 동일방식으로 데이터가 나뉘도록하여 실험결과고정.(숫자42 관례적사용)

#3.결측치,이상치없음
#4.스케일링 또는 표준화, 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #비지도학습 fit -> transtorm
X_test_scaled = scaler.fit_transform(X_test) #비지도학습 fit -> transtorm
# •StandardScaler(): 각특성단위 통일하는 표준화도구. 각특성평균0, 표준편차1로 변환해, 소득(MedInc)처럼 값범위큰특성이 모델에과도한영향을 미치는것을 방지
# •scaler.fit_transform(X_train): 훈련데이터에 맞춰 스케일러학습(fit), 그규칙에 따라 데이터변환(transform).
# •scaler.transform(X_test):⭐테스트데이터는 훈련데이터에서 학습된 스케일링규칙적용해 변환(transform)만 함. 모델이 실전에서 처음보는데이터 처리하는상황 시뮬레이션하기 위함.
# ⚠️참고: 제공된 원본코드의 scaler.fit_transform(X_test)는 테스트데이터에 스케일러를 새로학습시키는것으로, 이는 데이터누수(Data Leakage)오류에 해당. 훈련세트의 보(평균,표준편차)로 테스트세트를 변환해야 올바른평가가능.

#5.선형회귀모델학습
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#6.테스트데이터로 예측
y_pred = model.predict(X_test_scaled)

#7.회귀모델 평가지표 - 실제값, 기대값=예측값
mae = mean_absolute_error(y_test, y_pred)
print("mae:평균절대오차", mae)

mse = mean_squared_error(y_test, y_pred)
print("mse:평균제곱오차", mse)

rmse = np.sqrt(mse)
print("rmse:평균제곱근오차", rmse)

#R2결정계수가 1에 가까울수록 굿, 특성개수많아지면 R2가 예측과상관없이 좋아지므로 쓰지마
#특성많아질경우: mae,mse사용. mse가 이상치에 영향多
r2 = r2_score(y_test, y_pred) #결정계수, 평상시 score함수와 동일
print("결정계수: ", r2)

print("모델의 score", model.score(X_test_scaled, y_test))

# <class 'sklearn.utils._bunch.Bunch'>
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25

# Index(['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
#        'Latitude', 'Longitude'],
#       dtype='object')
# 0    4.526
# 1    3.585
# 2    3.521
# 3    3.413
# 4    3.422
# 5    2.697
# 6    2.992
# 7    2.414
# 8    2.267
# 9    2.611
# Name: houseval, dtype: float64
# mae:평균절대오차 0.5353166913867702
# mse:평균제곱오차 0.5388649914036732
# rmse:평균제곱근오차 0.7340742410708014
# 결정계수:  0.5887810596909611
# 모델의 score 0.5887810596909611

#8.특성개수적어서 산포도행렬→ 
print("데이터개수")
print(X.shape)

X["hoisingval"] = y
print(X.head())
#데이터2만개이상. 차트그리고 데이터샘플링
df_sample = X.sample(n=2000, random_state=42) #원하는만큼 데이터샘플링
print(df_sample.shape)

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df_sample,
             diag_kind='kde',
             kind='scatter')
plt.show()
# 데이터개수
# (20640, 8)
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  hoisingval
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23       4.526
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22       3.585
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24       3.521
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25       3.413
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25       3.422
# (2000, 9)