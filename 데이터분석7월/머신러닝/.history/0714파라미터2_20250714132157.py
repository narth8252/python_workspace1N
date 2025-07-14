import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC #하이퍼파라미터 개수가많아서 해보자
from sklearn.metrics import classification_report, accuracy_score
import mglearn
import os

#classification_report: 분류중에서도 이진분류 평가 라이브러리
#accuracy_score: 단순히 정확도 판단기준
#GridSearchCV: 파라미터를 주면 각 파라미터별로 전체조합을 만들어서 다 돌려본다.
#
iris = load_breast_cancer()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)
#파라미터배열로 만들어 설정
param_grid={
    'svc':{
        'C':[0.1,1,10,100], #내부적으로 오차허용범위조절, 내부적으로 오차조율하는데 이값범위내에서 움직임.
                                #이값크면 과대적합 위험
        'gamma':[1, 0.1, 0.01, 0.001], #커널의 영향범위, 클수록 과대적합
        'kernel':['rbf', 'linear'] #비선형(데이터가 비선형일때 좋음), 선형구조(데이터가 선형일때)
    },
    'random_forest':{
        'n_estimator':[50, 100, 200], #트리를 여러개만든다. 50,100,200개
        'max_depth':[None, 3, 10, 20], #최대깊이
        'learning_rate':]0.01, 0.1, 0.2]
    }
           }

# svc = SVC()
svc = SVC(C=10, gamma=1, kernel='linear')

#직접학습하지않고 GridSearchCV에게 맡긴다
# grid = GridSearchCV(estimator=svc, param_grid=param_grid,
#                     cv=5, verbose=2, scoring='accuracy')
# #estimator - 학습모델 객체전달
# #param_grid - 파라미터로 쓸 대상
# #cv - kfold검증, 데이터가 충분히 많으면 10까지 가능
# #verbose - 출력로그 0:없음, 1:간단, 2:자세히
# #sccoring - 평가수단을 정확도에 맞춤
# grid.fit(X_train, y_train)
# print("최적의 파라미터")
# print(grid.best_params_)
# print("최고스코어")
# print(grid.best_score_)
"""
.....
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time=   0.0s
[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   5.1s
[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   6.9s
[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   2.8s
[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   6.6s
[CV] END ..................C=100, gamma=0.001, kernel=linear; total time=   6.0s
최적의 파라미터
{'C': 10, 'gamma': 1, 'kernel': 'linear'}
최고스코어
0.955403556771546
"""


# svc = grid.best_estimator_ #학습해놓은모델 가져와서
svc.fit(X_train, y_train)
print("훈련셋 평가: ", svc.score(X_train, y_train))
print("테스트셋 평가: ", svc.score(X_test, y_test))
"""
iris 97% 너무높음
훈련셋 평가:  0.9732142857142857
테스트셋 평가:  0.9736842105263158

load_breast_cancer 91%
훈련셋 평가:  0.9154929577464789
테스트셋 평가:  0.916083916083916
"""
"""
svc = SVC(C=10, gamma=1, kernel='linear')
svc.fit(X_train, y_train)
print("훈련셋 평가: ", svc.score(X_train, y_train))
print("테스트셋 평가: ", svc.score(X_test, y_test))
훈련셋 평가:  0.9788732394366197
테스트셋 평가:  0.9440559440559441
"""