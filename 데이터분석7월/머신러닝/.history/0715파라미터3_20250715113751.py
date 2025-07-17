import pandas as pd
import numpy as np
import optuna
import mglearn
import os
from sklearn.datasets import load_iris, load_breast_cancer #"분류 Classiri"
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC #하이퍼파라미터 개수가많아서 해보자
from sklearn.metrics import classification_report, accuracy_score

# classification_report : 분류중에서도 이진분류 평가 라이브러리
# accuracy_score : 단순히 정확도 판단기준
# GridSearchCV : 파라미터를 주면 각 파라미터 별로 전체 조합을 만들어서 다 돌려본다.

iris = load_breast_cancer()
#iris.data, iris.target => 데이터프레임으로
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# iris.target #0이 악성, 1이 양성 -> 둘이 값을 반전하자. 나중에 모델 평가 해석시에 그게 더 편함
y_change = np.where(iris.target==0, 1, 0) # 0->1 로 1-> 0으로 바꿈

#특성의 개수(문법공부)
# print(np.unique(y_change, return_counts=True))
# #return_counts=True는 value_counts역할
# #(array([0, 1]), array([357, 212]))
# print(*np.unique(y_change, return_counts=True)) #*은 언팩
# #[0 1] [357 212]
# print(zip(*np.unique(y_change, return_counts=True))) #zip으로 묶으면 튜플로묶임
# #(0, 352) (1, 212) <zip object at 0x0000026098FE5580>
# print(dict(zip(*np.unique(y_change, return_counts=True)))) #최종: dict으로 묶으려고 한것임
# # {np.int64(0): np.int64(357), np.int64(1): np.int64(212)}

print("---------유방암데이터셋 정보----------")
print(f"특성개수 {X.shape[1]}")
print(f"샘플개수 {X.shape[0]}")
print(f"클래스분포 (0:양성 1:악성) {dict(zip(*np.unique(y_change, return_counts=True) ))} ")
# 특성개수 30
# 샘플개수 569
# 클래스분포 (0:양성 1:악성) {np.int64(0): np.int64(357), np.int64(1): np.int64(212)}

#악성인사람과 양성인사람간의 데이터불균형, iris는 균형데이터3:3:3
#불균형데이터셋일 경우 훈련셋과 테스트셋을 쪼갤때 그 균형을 유지하면서 쪼개라
# stratify=y_change
X_train, X_test, y_train, y_test = train_test_split(X, y_change, random_state=1234,
                                                    test_size=0.2, stratify=y_change)

print("---------훈련데이터셋--------")
print(X_train.shape, X_test.shape)
print(f"y_train분포: {dict(zip(*np.unique(y_train, return_counts=True)))} ")
print(f"y_test분포 : {dict(zip(*np.unique(y_test, return_counts=True)))} ")

# (455, 30) (114, 30)
# y_train분포: {np.int64(0): np.int64(357), np.int64(1): np.int64(212)}
# y_test분포 : {np.int64(0): np.int64(357), np.int64(1): np.int64(212)}

#라이브러리 경향: 콜백함수 만들어서 파라미터로 전달
#optuna가 호출할 콜백함수 만들어야 한다
from sklearn.ensemble import RandomForestClassifier
def objective(trial): #변수명은 마음대로
    #optuna를 통해 탐색할 하이퍼파라미터 범위 정의
    #그리드서치 5,6,7,8,9,...20 ->시작,엔딩
    max_depth = trial.suggest_int('max_depth', 5, 20)  # 트리의 최대깊이
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  # 리프노드가 되기위한 최소샘플수
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  # 분할을 위한 최소 샘플 수 (최소 2 권장)
    n_estimators = trial.suggest_int('n_estimators', 10, 100)  # 트리 개수 범위 확장

    # 모델에 파라미터 넣기
    model = RandomForestClassifier(max_depth=max_depth, 
                                   min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split,
                                   n_estimators=n_estimators,
                                   random_state=42,
                                   n_jobs=-1)  # 내부프로세스-1고정: 모든 CPU 코어 사용
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # 예측정확도
    return accuracy  # 반드시 마지막에 리턴. 목적값

# optuna 스터디생성
study = optuna.create_study(direction='maximize')  # 정확도 최대화 방향
print("-----옵투나 최적화시작(50회 시도)-----")
study.optimize(objective, n_trials=50)  # 함수명 오류 수정: object -> objective
print(f"최고정확도: {study.best_trial.value}")
print(f"하이퍼파라미터: {study.best_trial.params}")

# 최적 파라미터로 최종 모델 학습 및 평가
best_params = study.best_trial.params
best_model = RandomForestClassifier(
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    n_estimators=best_params['n_estimators'],
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)
y_pred_final = best_model.predict(X_test)

print("\n----- 최종 모델 평가 -----")
print(classification_report(y_test, y_pred_final))
print(f"정확도: {accuracy_score(y_test, y_pred_final):.4f}")

# 특성 중요도 확인
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n----- 상위 10개 중요 특성 -----")
print(feature_importance.head(10))


