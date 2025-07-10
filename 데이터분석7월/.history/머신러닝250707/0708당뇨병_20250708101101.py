#250708 am9:20 회귀분석 복습.당뇨병
#당뇨병과 관련된 요소들이 있음, 1년뒤 값 예측
#알고리즘 중에 Knn이웃,의사결정트리,랜덤포레스트 등 몇개는 분류,회귀 모두 지원
#Lasso, Ridge
from sklearn.datasets import load_diabetes
#bunch라는 클래스타입으로 정리해서 주고, 
# 이상치,누락치,정규화까지 된 자료를 줌(이게 우리주업무) -pandas, numpy
data = load_diabetes()

#로지스틱회귀분류
print(data.keys())
print(data['target'][:10]) #[:10]10개만출력
print(data['data'][:10])
print(data['DESCR'])

X = data["data"]  #ndarray2차원배열, 현재10개의 특성값이
y = data["target"]#ndarray1차원배열, 미래예측값으로 나타남

print(X.shape) #데이터442개, 특성10개
print(y.shape)

#데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234) 
#튜플4개, 같은비율로 잘라내려고, test size=안주면 7.5:2.5비율로 자동나눔
#긴코딩 한줄넘어가서 엔터쳤더니 빨간줄 뜨면 끊은부분에 역슬래시\ 써

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train) #학습하고
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
print("<< 리니어레그레션 모델 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
# << 리니어레그레션 모델 결과 >>
# 훈련셋 점수 (R^2):  0.5303797620193524 
# 테스트셋 점수 (R^2):  0.4693760785596738
# 훈련셋과 테스트셋 점수차 적다면, 과적합(overfitting) 적은편.
#선형회귀모델의 score함수 썼을때 결정계수1이면 완벽예측, 0이면예측불가(심각하게 안맞음)

from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1)
model.fit(X_train, y_train) #학습하고
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
print("<< 릿지 모델 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
# << 릿지 모델 결과 >>
# 훈련셋 점수 (R^2):  0.5225441395914145
# 테스트셋 점수 (R^2):  0.4722580072140735
# 훈련셋과 테스트셋 점수차 적다면, 과대적합(overfitting) 리니어모델에 비해 적은편.

from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train) #학습하고
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
print("<< 라쏘 모델 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
# << 라쏘 모델 결과 >>
# 훈련셋 점수 (R^2):  0.5201725255864416
# 테스트셋 점수 (R^2):  0.4751660650991202
# 훈련셋과 테스트셋 점수차 적다면, 과대적합(overfitting) 리지모델에 비해 적은편.

#의사결정트리: 훈련셋100%로 언제나 과대적합, 특성의 중요도만 보기때문에 기울기랑 절편없음
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train) #학습하고
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
print("<< DecisionTreeRegressor Model 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
print("특성의 중요도: ", model.feature_importances_)
# << 의사결정트리 모델 결과 >>
# 훈련셋 점수 (R^2):  1.0
# 테스트셋 점수 (R^2):  -0.17834691111562684 (결정계수가 음수값나오면 못쓰는것임)
# 특성의 중요도:  [0.04210206 0.01931368 0.37496183 0.07400458 0.02517021 0.10012291
#  0.0431841  0.02747952 0.18676855 0.10689257]
# 회귀분석에서 score가 결정계수값나오는데 음수면 위험.

#랜덤포레스트(앙상블) = 의사결정트리+업그레이드, 여러개의 분석기를 함께사용하는 앙상블
from sklearn.e import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train) #학습하고
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
print("<< DecisionTreeRegressor Model 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
print("특성의 중요도: ", model.feature_importances_)


# R² (결정계수, Coefficient of Determination) 회귀모델 예측측정 대표적인지표
# 공식: R² = 1-(잔차제곱합(SSR)/전체제곱합(SST)) = 설명된분산/전체분산 = 1-(못맞춘분산/전체분산)
# SST(TotalSumofSquares): 실제 값이 평균으로부터 얼마나 떨어져 있는가 (분산)
#                         아무 모델 없이 평균만 썼을 때의 오차
# SSR(ResidualSumofSquares): 예측값과 실제값 사이의 차이 (오차)
#                             모델을 써서 예측한 후의 오차"

# R² = 1: 예측 완벽 (오차없음)모두 맞췄다
# R² = 0: 기본평균만큼 = 예측의미없음
# R² < 0: 모델이 평균만도 못맞춤 → 나쁜 모델
#선형회귀모델의 score함수 썼을때 결정계수1이면 완벽예측, 0이면예측불가(심각하게 안맞음)
# 훈련셋과 테스트셋 점수차 적다면, 과적합(overfitting) 적은편.

#로지스틱도 가중치가 있음. 왜냐면 따라서 만들었으므로
# print(model.coef_)
# print(model.intercept_)
# print(X_train.shape)
# print(X_test.shape)
