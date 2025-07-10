#sklearn에서 load_digits() → 손으로쓴숫자 맞추기
#애초에 미국우편번호 나누기위해 개발돼 그때 수집된 데이터
#이미지 → 디지털화하는 과정에 흑백은2차원배열, 컬러는3차원배열임
#이미지가 10장있고 각이미지크기가 150 by 150
#10 150x150이 특성개수가 된다. 이미지를 읽어서 → numpy배열로 변경(오래걸리는데 파이썬PIL라이브러리로 제공)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1.데이터준비:현재데이터는 정리해서 준 데이터이지만, 실제데이터는 우리가 이 작업해야함
data = load_digits()    
X = data["data"]
y = data["target"]
print(X.shape)
print(X[:10])

print("-------이미지 key값 추가돼서 그림으로 그려줌-------")
print( data.images[:10]) #numpy 2D → 1D로 바꿔 data로 준거고 원래데이터
images = data.images
# #이미지1개출력
# plt.figure(figsize=(10, 4)) #차트의 크기
# plt.imshow(images[0], cmap="gray_r") #gray로 이미지출력
# plt.show()

#이미지여러개 동시출력하려면 화면분할(inch단위)
def drawNumbers():
    plt.figure(figsize=(10, 4)) #화면전체크기 지정후 작게 나눌시 subplot함수사용
    # 2 by 5 로 쪼개면 10개의 화면만들어지고 각분할위치에 번호인덱스 붙음
    # 0 1 2 3 4 
    # 5 6 7 8 9
    for i in range(10):
        plt.subplot(2, 5, i+1) #내가 내보낼 위치지정
        plt.imshow(images[i], cmap="gray_r", interpolation='nearest') #없으면 옆색깔써 보간법
        plt.title(f"Label:{y[i]}")
        plt.axis('off') #축없애기

    plt.tight_layout() #이쁘게 다시 정리해라
    plt.suptitle("first 10 Digits images", y=1.05, fontsize=16) #한글비추,영어로 써라
    #y는 제목이 출력될위치, y=0아래쪽, y=1위쪽, y=1.05영역밖에 놓아라.
    plt.show()
# (1797, 64) → (1797장의 이미지, 8by8 이미지크기) → 1차원으로 바꾸니까 64개의 특성이 되고, 

#데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

#로지스틱 분류분석
from sklearn.linear_model import LogisticRegression #이름은 회귀인데 분류가 맞음
# solver 
model = LogisticRegression(solver='liblinear', #모델계수 찾아가는법(데이터셋적을때'liblinear')
                           multi_class='auto', #다중분류시
                           max_iter=5000, #계수찾아갈때 반복학습회수
                           random_state=0)
model.fit(X_train, y_train)
print("=== 로지스틱 분류분석 ===")
print("훈련셋:", model.score(X_train, y_train))
print("테스트셋:", model.score(X_test, y_test)) #회귀분석에서 score가 결정계수값이 나오는데 음수면 위험
# === 로지스틱 분류분석 ===
# 훈련셋: 0.9936356404136834 #아주높은성과
# 테스트셋: 0.9629629629629629

#KNN이웃: "가까운 데이터끼리 비슷한 특성을 가진다" 는 가정으로 데이터작고 직관적분포에 추천
# K가 너무작으면 과적합, 너무크면 과소적합
# 새데이터 포인트가 들어오면, 주변의 K개의 이웃을 찾아보고
# 분류: 가장 많은 클래스를 선택
# 회귀: 이웃들의 평균값을 계산
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3) #이웃개수3개, 바꿔가며 돌려봐야함.
model.fit(X_train, y_train)
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
print("=== KNN이웃 분석 ===")
print("훈련셋평가", model.score(X_train, y_train))
print("테스트셋평가", model.score(X_test, y_test))
# === KNN이웃 분석 ===
# 훈련셋평가 0.9888623707239459
# 테스트셋평가 0.9888888888888889


#의사결정트리: 훈련셋100%로 언제나 과대적합, 특성의 중요도만 보기때문에 기울기랑 절편없음
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train) #학습하고
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
print("<< DecisionTreeClassifier Model 결과 >>")
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
# 랜덤포레스트: 의사결정트리 + 업그레이드, 여러 개의 분석기를 함께 사용 - 앙상블
from sklearn.ensemble import RandomForestClassifier
# 트리를 랜덤으로 계속만들어내서 측정값달라지므로 
#                               0으로 고정필수,만들트리의 최대개수, 트리최대깊이 지정,  학습속도
model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=5)
model.fit(X_train, y_train)     # 학습을 하고
y_pred = model.predict(X_test)
print("=== RandomForestClassifier Model ===")
print("훈련셋:", model.score(X_train, y_train))
print("테스트셋:", model.score(X_test, y_test))
print("특성의 중요도:", model.feature_importances_)
# === RandomForestRegressor Model ===
# 훈련셋: 0.5844439277238154
# 테스트셋: 0.5656300129402022

# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
# 트리를 랜덤으로 계속만들어내서 측정값달라지므로 
#                               0으로 고정필수,만들트리의 최대개수, 트리최대깊이 지정,  학습속도
model = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)     # 학습을 하고
y_pred = model.predict(X_test)
print("=== GradientBoostingRegressor Model ===")
print("훈련셋:", model.score(X_train, y_train))
print("테스트셋:", model.score(X_test, y_test))
# === GradientBoostingRegressor Model ===
# 훈련셋: 0.9737250675620381
# 테스트셋: 0.8494497305560165

