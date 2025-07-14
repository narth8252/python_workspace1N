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
pram_grid={'C':[0.1,1,10,100], #내부적으로 오차허용범위조절, 내부적으로 오차조율하는데 이값크면 괒
           }

svc = SVC()


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