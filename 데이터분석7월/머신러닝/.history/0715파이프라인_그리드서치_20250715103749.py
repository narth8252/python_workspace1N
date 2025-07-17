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


#파라미터배열로 만들어 설정
param_grid={'C':[0.1,1,10,100], #내부적으로 오차허용범위조절, 내부적으로 오차조율하는데 이값범위내에서 움직임.
                                #이값크면 과대적합 위험
            'gamma':[1, 0.1, 0.01, 0.001], #커널의 영향범위, 클수록 과대적합
            'kernel':['rbf', 'linear'] #비선형(데이터가 비선형일때 좋음), 선형구조(데이터가 선형일때)
           }
svc = SVC(C=10, gamma=1, kernel='linear')
