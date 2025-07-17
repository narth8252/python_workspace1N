import pandas as pd
import numpy as np
import optuna
from sklearn.datasets import fetch_california_housing #"회귀Regression문제" 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

#California Housing 데이터셋에 "회귀문제"를 Optuna 사용해 최적화,모델평가,시각화 → 모델성능,특성 분석
#주택 가격(연속값)을 예측하는 회귀(Regression) 문제
# 회귀문제에 적합한 평가지표: RMSE(Root Mean Squared Error), MAE(Mean Absolute Error), R²(결정계수) 등
#RMSE를 사용한 구체적인 이유:
# 1. 연속값 예측 평가:
#  • Accuracy는 예측이 정확히 맞았는지(1) 틀렸는지(0)만 평가하므로 연속값 예측에는 부적합합니다.
#  • RMSE는 예측값과 실제값 사이의 차이(오차)를 직접 측정하므로 회귀 문제에 적합합니다.
# 2. 오차의 크기 반영:
#  • RMSE는 오차를 제곱하기 때문에 큰 오차에 더 큰 페널티를 부여합니다.
#  • 주택 가격과 같은 중요한 예측에서는 큰 오차를 줄이는 것이 중요합니다.
# 3. 원래 스케일로 해석 가능:
#  • RMSE는 원래 타겟 변수(주택 가격)와 같은 단위로 표현되어 직관적으로 해석이 가능합니다.
#  • 예: RMSE가 0.5라면, 평균적으로 예측 주택 가격이 실제 가격과 0.5 단위 차이가 난다는 의미입니다.
# 회귀 문제에서 자주 사용하는 다른 평가 지표들:
# 1. MAE(Mean Absolute Error):
#  • 절대 오차의 평균으로, 이상치에 덜 민감합니다.
#  • RMSE보다 해석이 더 직관적일 수 있습니다.
# 2. R²(결정계수):
#  • 모델이 설명할 수 있는 타겟 변수 분산의 비율을 나타냅니다.
#  • 0~1 사이의 값으로, 1에 가까울수록 모델 성능이 좋음을 의미합니다.

#코딩순서:
# 1. 데이터 로딩 및 탐색:
#  • California Housing 데이터셋을 로드하고 기본 정보 확인
# 데이터를 훈련/테스트 세트로 분할 (80/20)
# Optuna 최적화:
# 회귀 문제이므로 RandomForestRegressor 사용
# 하이퍼파라미터: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
# 목표 지표: RMSE(Root Mean Squared Error) 최소화
# 최적 모델 평가:
# RMSE, MAE(Mean Absolute Error), R²(결정계수) 계산
# 최적 모델의 성능 평가 및 특성 중요도 분석
# 시각화:
# 특성 중요도 시각화: 어떤 특성이 예측에 가장 중요한지 파악
# 실제값 vs 예측값: 모델이 얼마나 정확하게 예측하는지 시각화
# 잔차 분석: 예측 오차의 분포와 패턴을 확인하여 모델의 한계점 파악

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
    max_features = trial.suggest_float('max_features', 0.3, 1.0)

    #모델에 파라미터 넣기
    model = RandomForestRegressor(max_depth=max_depth, 
                                  max_features=max_features, 
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  n_estimators=n_estimators,
                                  random_state=42,
                                  n_jobs=-1) #내부프로세스-1고정:CPU개수*2라서 알아서 최대치사용
     
    model.fit(X_train, y_train) # 모델 학습
    y_pred = model.predict(X_test)  # 예측
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




