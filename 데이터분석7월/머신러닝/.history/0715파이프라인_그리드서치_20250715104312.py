import pandas as pd
import numpy as np
import mglearn
import os
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC #하이퍼파라미터 개수가많아서 해보자
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline  # 여러 단계의 처리 과정을 순차적으로 연결하는 파이프라인
from sklearn.preprocessing import StandardScaler  # 데이터 표준화(평균 0, 분산 1)를 위한 클래스
from sklearn.linear_model import LogisticRegression  

#classification_report: 분류중에서도 이진분류 평가 라이브러리
#accuracy_score: 단순히 정확도 판단기준
#GridSearchCV: 파라미터를 주면 각 파라미터별로 전체조합을 만들어서 다 돌려본다.

iris = load_breast_cancer()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

#1.파이프라인 구축
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # 1단계: 데이터 스케일링 (평균 0, 분산 1로 표준화)
    ('classifier', LogisticRegression(random_state=42)),  # 1단계: 데이터 스케일링 (평균 0, 분산 1로 표준화)
])

#2.그리드서치 구축
#파라미터배열로 만들어 설정
param_grid={
    'scaler':[StandardScaler(), MinMaxScalar()],
    'classifier_C':[0.01, 0.1, 10, 100],
    'classifier_solver':['liblinear', 'lbfgs']
}

#3.그리드서치 만들기
from sklearn.model_selection import StratifiedKFold
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv = StratifiedKFold()
)    #cv에 숫자5 -Kfold검증, 타겟이
# svc = SVC(C=10, gamma=1, kernel='linear')
