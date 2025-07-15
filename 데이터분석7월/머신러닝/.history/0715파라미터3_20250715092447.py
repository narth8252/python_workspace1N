import pandas as pd
import numpy as np
import optuna
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC #하이퍼파라미터 개수가많아서 해보자
from sklearn.metrics import classification_report, accuracy_score
import mglearn
import os

# classification_report : 분류중에서도 이진분류 평가 라이브러리
# accuracy_score : 단순히 정확도 판단기준
# GridSearchCV : 파라미터를 주면 각 파라미터 별로 전체 조합을 만들어서 다 돌려본다.
#
iris = load_breast_cancer()
#iris.data, iris.target => 데이터프레임으로
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# iris.target #0이 악성, 1이 양성 -> 둘이 값을 반전하자. 나중에 모델 평가 해석시에 그게 더 편함
y_change = np.where(iris.target==0, 1, 0) # 0->1 로 1->0으로 바꿈

#특성의 개수
print(np.unique(y_change, return_counts=True))
#return_counts=True는 value_counts역할
print(*np.unique(y_change, return_counts=True)) #*은 언팩


X_train, X_test, y_train, y_test = train_test_split(X, y_change, random_state=1234)







