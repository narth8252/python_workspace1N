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
print("y.shape:", y.shape)
print(y.shape) #디버깅용print → 행개수 동일해야 연산수행, 많은쪽 잘라내고 작업
#D가 흑인비율이라 인종차별문제로 이 데이터는 이제 안씀.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression() 
model.fit(X_train, y_train)
print("LinearRegression")
print("훈련셋: ", model.score(X_train, y_train))
print("테스트셋: ", model.score(X_test, y_test))
print("기울기들: ", model.coef_)
print("절편: ", model.intercept_)

#선형회귀분석: 다중공선성문제, 여러특성간 서로 너무 밀접해 필요없는 요소들을 고려X
#특성개수 많을때 처리능력 저하
# → 개선된게 라쏘, 리지 알고리즘
#라쏘: 쓸데없는 계수(기울기)

