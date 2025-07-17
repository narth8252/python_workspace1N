import pandas as pd
import numpy as np
import optuna
import mglearn
import os
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC #하이퍼파라미터 개수가많아서 해보자
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline  # 여러 단계의 처리 과정을 순차적으로 연결하는 파이프라인
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

iris = load_breast_cancer()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_change = np.where(iris.target==0, 1, 0) # 0->1 로 1->0으로 바꿈


print("---------유방암데이터셋 정보----------")
print(f"특성개수 {X.shape[1]}")
print(f"샘플개수 {X.shape[0]}")
print(f"클래스분포 (0:양성 1:악성) {dict(zip(*np.unique(y_change, return_counts=True) ))} ")
X_train, X_test, y_train, y_test = train_test_split(X, y_change, random_state=1234,
                                                    test_size=0.2, stratify=y_change)

print("---------훈련데이터셋--------")
print(X_train.shape, X_test.shape)
print(f"y_train분포: {dict(zip(*np.unique(y_change, return_counts=True) ))} ")
print(f"y_test분포 : {dict(zip(*np.unique(y_change, return_counts=True) ))} ")
# (455, 30) (114, 30)

#라이브러리 경향: 콜백함수 만들어서 파라미터로 전달
#optuna가 호출할 콜백함수 만들어야 한다
def objective(trial): #변수명은 마음대로
    #optuna를 통해 탐색할 하이퍼파라미터 범위 정의
    max_depth = trial.suggest_int('max_depth', 5, 20) #그리드서치 5,6,7,8,9,...20 ->시작,엔딩
    #트리의 최대깊이
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    #리프노드가 되기위한 최소샘플수
    min_samples_split = trial.suggest_int('min_samples_split', 1, 10)
    n_estimators = trial.suggest_int('n_estimators', 1, 10)

    #모델에 파라미터 넣기
    model = RandomForestClassifier(max_depth=max_depth, 
                                   min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split,
                                   n_estimators=n_estimators,
                                   random_state=42,
                                   n_jobs=-1) #내부프로세스-1고정:CPU개수*2라서 알아서 최대치사용
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) #예측정확도
    return accuracy #반드시 마지막에 리턴. 목적값

    #파이프라인 구축
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 1단계: 데이터 스케일링 (평균 0, 분산 1로 표준화)
        ('classifier', model)
    ])

    # 학습 데이터로 모델 학습 (파이프라인 적용)
    pipeline.fit(X_train, y_train)
    # 테스트 데이터에 대한 예측 수행
    y_pred = pipeline.predict(X_test)
    # 모델 성능 평가 및 결과 출력
    # classification_report: 정확도, 정밀도, 재현율, F1 점수 등 다양한 평가 지표를 한번에 보여줌
    print(classification_report(y_test, y_pred))

#optuna 스터디생성
study = optuna.create_study(direction='maximize') #mode=RandomForest()
#이익최대화 방향으로 study객체 만든다
print("-----옵투나 최적화시작(50회 시도)-----")
study.optimize(object, n_trials=50) #콜백함수, 회수지정
#optimize - 최적화함수
print(f"최고정확도: {study.best_trial.value}")
print(f"하이퍼파라미터: {study.best_trial.params}")






