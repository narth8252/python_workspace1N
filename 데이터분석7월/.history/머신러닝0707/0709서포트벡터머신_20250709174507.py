# 데이터 스케일링(Scaling)이 서포트벡터머신(SVM) 모델의 성능에 얼마나 중요한 영향을 미치는지" 직접 보여주는 것

# 1. 데이터 준비하고 문제지X와 정답지y로 나누기
from sklearn.datasets import load_breast_cancer

# sklearn라이브러리에 내장된 유방암진단데이터 불러오기
cancer = load_breast_cancer()
print(cancer.keys())
# X=종양크기,질감 등 진단에 사용된 여러측정값(특성feature)을 저장. 이게 문제지
# y=각종양이 악성(0)인지 양성(1)인지 나타내는 정답지 y
X = cancer['data']
y = cancer['target']

# 2. 스케일링전 모델성능확인 : 가공전원본으로 학습시켜 성능을 확인하고 기준점잡기
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
ss = StandardScaler()       # 객체 생성
X_scaled = ss.fit_transform(X)    # 학습하고 바로 변경된 값 반환
print(X_scaled)

# 2-1. train_test_split: 원본데이터의X,y를 학습용(train)과 테스트용(test)으로 나눠
#                        모델은 학습용데이터로 공부하고, 테스트용데이터로 시험봄
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

"""
서포트벡터머신
대부분의 머신러닝 알고리즘은 평면에 선을 긋는다.
데이터에 따라서는 평면에 선을 못 긋는 경우에
수학자 차원을 분리시켜서 평면의 다차원공간으로 보내서 차원간에 선을 긋는다.
"""
# 2-2.모델학습 및 평가
# 로지스틱회귀(비교용):기본적인 분류모델인 로지스틱으로 학습후 성능측정.
#                   이 모델은 스케일링의 영향을 덜받아 비교기준으로 적합
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("-------- 로지스틱 회귀(비교용) ----------")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

# 
from sklearn.svm import SVC
model = SVC()   # 분류
model.fit(X_train, y_train)
print("-------- 스케일링 안한 서포트벡터머신 ----------")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

print()
print("-------- 스케일링된 서포트벡터머신 ----------")
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y, random_state=0)
model.fit(X_train_scaled, y_train_scaled)
print("훈련셋", model.score(X_train_scaled, y_train_scaled))
print("테스트셋", model.score(X_test_scaled, y_test_scaled))





