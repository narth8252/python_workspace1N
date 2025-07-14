# 250707 pm2:10 250701딥러닝_백현숙.PPT-285p
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-p
# 저장폴더 C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707

#다중회귀분석: 공분산을 따져보고 특성제거해야함
#R은 기본적으로 제거하지만, 파이썬은 내가 해야함.
#powershell은 파일명에()를 못읽음

import pandas as pd #다양한유형의 데이터있을때 처리방법
import numpy as np

# 1. 데이터 불러오기
url = "http://lib.stat.cmu.edu/datasets/boston"
#분기문자가 공백이 아니고
df = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
print(df.head(10))

# 2. 특성과 타겟 분리
# 데이터 정리 (두 줄마다 병합해서 13개 특성 만듦)
#np에 hstack함수있음. 수평방향으로 배열 이어붙이는 함수
#짝수행에 홀수 갖다붙임 df.values[::2, :] → 0 2 4 6 8 : 전체컬럼
#홀수행의 앞열2개만 df.values[1::2, :2]
data = np.hstack([df.values[::2, :], df.values[1::2, :2]])  # (506, 13)
print(data[:10]) #넘파이 배열의 장점 → 뒤에 붙일수있음
X = data
y = df.values[1::2, 2]  # (506,) #이열이 target
print("X.shape:", X.shape)
print("y.shape:", y.shape) #디버깅용print → 행개수 동일해야 연산수행, 많은쪽 잘라내고 작업
#D가 흑인비율이라 인종차별문제로 이 데이터는 이제 안씀.

# 3. 훈련/테스트 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 4. 선형회귀 모델 훈련
from sklearn.linear_model import LinearRegression
model = LinearRegression() 
model.fit(X_train, y_train)

# 5. 결과 출력
print("<< Linear Regression 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
print("기울기(coef_): ", model.coef_)
print("절편(intercept_)", model.intercept_)

# 변수	     설명
# df	    원본 DataFrame (텍스트 불러온 원시 형태)
# data → X	특성 데이터 13개 (열 재정렬해서 병합)
# y      	집값(MEDV), df의 2번째 줄의 세 번째 열 값
# X.shape, y.shape	모두 (506, …) 이면 OK
# LinearRegression().score()	R² 점수 (1에 가까울수록 설명력 좋음)

#선형회귀분석: 다중공선성문제, 여러특성간 서로 너무 밀접해 필요없는 요소들을 고려X
#특성개수 많을때 처리능력 저하 → 개선된게 라쏘, 리지 알고리즘

#보스톤 주택가격데이터의 특성개수는 13개라 가중치도13개
#누군가 가중치 규제하면 과대적합 막을까?
#가중치를 규제하자 → 라쏘는 가중치를 0에 가깝게하다가 불필요한요소있으면 아예0으로 만들기도.
#모델을 심플하게 만든다.

#라쏘: 쓸데없는 계수(기울기)

# 6. 리지 모델 훈련
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1) #0에 가까워지더라도 0은안됨.
model.fit(X_train, y_train)
print("<< Ridge 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
print("기울기(coef_): ", model.coef_)
print("절편(intercept_)", model.intercept_)

# 7. 라쏘 모델 훈련
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1) #0.1 → 10으로 숫자커질수록 규제커짐
model.fit(X_train, y_train)
print("<< Lasso 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
print("기울기(coef_): ", model.coef_)
print("절편(intercept_)", model.intercept_)

# R² (결정계수, Coefficient of Determination) 회귀모델 예측측정 대표적인지표
# 공식: R² = 1-(잔차제곱합(SSR)/전체제곱합(SST))
# SST(TotalSumofSquares): 실제 값이 평균으로부터 얼마나 떨어져 있는가 (분산)
#                         아무 모델 없이 평균만 썼을 때의 오차
# SSR(ResidualSumofSquares): 예측값과 실제값 사이의 차이 (오차)
#                            모델을 써서 예측한 후의 오차"

📊 쉽게 말하면?
상황	R² 값	해석
모델이 모두 맞췄다	1	완벽한 예측
모델이 평균값만큼 예측	0	예측이 의미 없음 (기본 평균만큼)
모델이 완전히 엉망	< 0	예측이 평균보다 더 못함