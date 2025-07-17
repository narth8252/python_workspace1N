import pandas as pd
import numpy as np
import optuna
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

#California Housing 데이터셋에 대한 회귀 문제를 Optuna를 사용하여 최적화하고, 자세한 모델 평가와 시각화를 수행합니다. 이를 통해 모델의 성능과 특성을 종합적으로 분석할 수 있습니다.

# California Housing 데이터셋 로드
print("----- California Housing 데이터셋 로드 중 -----")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print("----- California Housing 데이터셋 정보 -----")
print(f"특성 개수: {X.shape[1]}")
print(f"샘플 개수: {X.shape[0]}")
print(f"타겟 변수(주택 가격) 통계:")
print(f"  - 최소값: {y.min():.2f}")
print(f"  - 최대값: {y.max():.2f}")
print(f"  - 평균값: {y.mean():.2f}")
print(f"  - 표준편차: {y.std():.2f}")

# 데이터셋 분할 (훈련, 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n----- 훈련/테스트 데이터셋 크기 -----")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
# print(f"y_train분포: {dict(zip(*np.unique(y_change, return_counts=True) ))} ")
# print(f"y_test분포 : {dict(zip(*np.unique(y_change, return_counts=True) ))} ")

# 데이터 간략히 살펴보기
print("\n----- 특성 미리보기 -----")
print(X.head())

# Optuna 최적화를 위한 콜백함수
def objective(trial): #변수명은 마음대로
    # Optuna를 통해 탐색할 하이퍼파라미터 범위 정의
    max_depth = trial.suggest_int('max_depth', 3, 25)  # 트리의 최대깊이
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  # 리프노드가 되기위한 최소샘플수
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  # 분할을 위한 최소 샘플 수
    n_estimators = trial.suggest_int('n_estimators', 50, 300)  # 트리의 개수 (범위 확장)

    #모델에 파라미터 넣기
    model = RandomForestRegressor(max_depth=max_depth, 
                                  max_depth=max_depth, 
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  n_estimators=n_estimators,
                                  random_state=42,
                                  n_jobs=-1) #내부프로세스-1고정:CPU개수*2라서 알아서 최대치사용
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) #예측정확도
    return accuracy #반드시 마지막에 리턴. 목적값

    # 파이프라인 구축
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 1단계: 데이터 스케일링 (평균 0, 분산 1로 표준화)
        ('classifier', model)  # 2단계: 분류 모델 적용
    ])

    # 학습 데이터로 모델 학습 (파이프라인 적용)
    pipeline.fit(X_train, y_train)
    # 테스트 데이터에 대한 예측 수행
    y_pred = pipeline.predict(X_test)
    # 모델 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy  # 최적화 목표값 반환

# Optuna 스터디 생성
study = optuna.create_study(direction='maximize')  # 정확도 최대화 방향으로 study 객체 생성

print("-----옵투나 최적화시작(50회 시도)-----")
study.optimize(objective, n_trials=50)  # 'object'를 'objective'로 수정, 50회 시도

print(f"최고정확도: {study.best_trial.value}")
print(f"최적 하이퍼파라미터: {study.best_trial.params}")

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

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', best_model)
])

best_pipeline.fit(X_train, y_train)
y_pred_final = best_pipeline.predict(X_test)

print("\n----- 최종 모델 평가 -----")
print(classification_report(y_test, y_pred_final))




