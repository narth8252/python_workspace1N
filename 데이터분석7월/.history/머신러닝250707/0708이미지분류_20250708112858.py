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

#Knn이웃, 의사결정트리, 랜덤포레스트, 그라디언트부스팅
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3) #이웃개수3개, 바꿔가며 돌려봐야함.
model.fit(X_train, y_train)

#예측하기
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
#본래 테스트셋인 y_test와 비교
print(y_pred)
print(y_test)

#평가하기
print("훈련셋평가", model.score(X_train, y_train))
print("테스트셋평가", model.score(X_test, y_test))

