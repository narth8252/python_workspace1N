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
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)
svc = SVC()
svc.fit(X_train, y_train)
print("훈련셋 평가: ", svc.score(X_train, y_train))