#0707 pm1시 윤년
#입력데이터[[1],[2],[3]...]
# 출력데이터 [0,0,0,1,0,0,0,1....]
#1~20205

def isLeap(year):
    if year%4==0 and year%100!=0 or year%400==0:
        return 1
    return 0

X = []  
y = []
#데이터 생성 조작할때는 리스트가 편함
for i in range(1, 2026):
    X.append(i)
    y.append(isLeap(i))
print(X[2000:])
print(y[2000:])

#머신러닝,X는 반드시 ndarray(2d), y는 1d어레이여야함

import numpy as np
X = n.array(X)

#나름 데이터 전처리하는 과정임
#(-1,1)하면 차원추가 → 2차원만들기
X = X.reshape(-1, 1)
print(X.shape) #2d array타입으로 만들어야 한다.
y = np.array(y)

#1.훈련셋과 테스트셋을 나누자
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
#위에 숫자맘대로. 6:4나 7:3
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3) #Knn이웃 분류알고리즘
model.fit(X_train, y_train) #X,y값으로 학습진행

#둘다 찍어서 점수확인해봐야 확신이듬
print("훈련셋: ")