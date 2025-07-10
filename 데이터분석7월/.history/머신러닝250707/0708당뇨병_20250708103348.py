#250708 am9:20 회귀분석 복습.당뇨병
#당뇨병과 관련된 요소들이 있음, 1년뒤 값 예측

#myenv1서버에 프로그램 깔기
#1. cmd 관리자권한실행(anaconda됨) 
#2. conda activate myenv1
#3. conda install numpy scipy scikit-learn matplotlib ipython pandas imageio pillow graphviz python-graphviz
#4. pip install xgboost lightgbm catboost 
#5. conda install Microsoft Visual C++ Build Tools (설치안되면MS가서 다운로드)
#깔리면 VS코드도 (상단재생버튼(Python)으로 안되면 posershell에나 cmd에서)
#6. conda activate myenv1


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
from sklearn.ensemble import GradientBoostingRegressor
# 트리를 랜덤으로 계속만들어내서 측정값달라지므로 
#                               0으로 고정필수,만들트리의 최대개수, 트리최대깊이 지정
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)     # 학습을 하고
y_pred = model.predict(X_test)
print("=== GradientBoostingRegressor Model ===")
print("훈련셋:", model.score(X_train, y_train))
print("테스트셋:", model.score(X
#트리계열은 특성의중요도 가지고다님
# << RandomForestRegressor Model 결과 >>
# 훈련셋 점수 (R^2):  0.6618828640833401 #n_estimators=100
# 테스트셋 점수 (R^2):  0.4747874939992528 #여전히 과대적합있지만 의사결정트리보단 나음.
# 특성의 중요도:  [0.02291191 0.00547266 0.4171142  0.07726188 0.01796369 0.02882855
#  0.04581483 0.02043824 0.28816204 0.076032]
# << RandomForestRegressor Model 결과 >>
# 훈련셋 점수 (R^2):  0.6627851578348634  #n_estimators=300
# 테스트셋 점수 (R^2):  0.4740713076615144
# 특성의 중요도:  [0.02696604 0.00471781 0.42519002 0.07228074 0.01700467 0.02792643
#  0.04552961 0.02293954 0.28250549 0.07493965]

#xgboost(앙상블): 약한학습기들(의사결정트리) 통해서 학습하고 보정작업거쳐 결과찾음
#sklearn GradientBoostion, xgboost라이브러리, lightGBM..
# learning_rate=0.1 학습속도(너무높으면 빨리하다가 최적위치 지나칠수있고, 너무낮으면 느린학습으로 최저점에 도달못할수도 있음.)
#GridSearch: 하이퍼파라미터들을 주면 알아서 테스트하면서 적절한 파라미터찾아 오래걸림
# 트리를 랜덤으로 계속만들어내서 측정값달라지므로 
#                               0으로 고정필수,만들트리의 최대개수, 트리최대깊이 지정, 학습속도
from xgboost import XGBRegressor
model = XGBRegressor(random_state=0, n_estimators=300, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)     # 학습을 하고
y_pred = model.predict(X_test)
print("=== XGBRegressor Model ===")
print("훈련셋:", model.score(X_train, y_train))
print("테스트셋:", model.score(X_test, y_test))
print("특성의 중요도:", model.feature_importances_)

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
