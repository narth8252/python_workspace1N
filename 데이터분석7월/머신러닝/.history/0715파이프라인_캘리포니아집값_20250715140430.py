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
#1. 데이터 로딩 및 탐색:
# • California Housing 데이터셋을 로드하고 기본 정보 확인
# • 데이터를 훈련/테스트 세트로 분할 (80/20)
#2. Optuna 최적화:
# • 회귀 문제이므로 RandomForestRegressor 사용
# • 하이퍼파라미터: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
# • 목표 지표: RMSE(Root Mean Squared Error) 최소화
#3. 최적 모델 평가:
# • RMSE, MAE(Mean Absolute Error), R²(결정계수) 계산
# • 최적 모델의 성능 평가 및 특성 중요도 분석
#4. 시각화:
# • 특성 중요도 시각화: 어떤 특성이 예측에 가장 중요한지 파악
# • 실제값 vs 예측값: 모델이 얼마나 정확하게 예측하는지 시각화
# • 잔차 분석: 예측 오차의 분포와 패턴을 확인하여 모델의 한계점 파악

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
     
    model.fit(X_train, y_train) #모델학습
    y_pred = model.predict(X_test)  #예측
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) #평가(RMSE-낮을수록 좋음)
    return rmse  #목표:RMSE최소화 

# Optuna 스터디 생성 - RMSE 최소화 방향으로 설정
print("\n----- Optuna 최적화 시작 (50회 시도) -----")
study = optuna.create_study(direction='minimize')  # RMSE를 최소화
study.optimize(objective, n_trials=50)

print(f"\n----- 최적화 결과 -----")
print(f"최소 RMSE: {study.best_trial.value:.4f}")
print(f"최적 하이퍼파라미터: {study.best_trial.params}")

# 최적 파라미터로 최종 모델 학습 및 평가
best_params = study.best_trial.params
best_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42,
    n_jobs=-1
)

# 최적 모델 학습
best_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = best_model.predict(X_test)

# 평가 지표 계산
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n----- 최종 모델 평가 -----")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 특성 중요도 시각화
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n----- 특성 중요도 -----")
print(feature_importance)

# 시각화: 특성 중요도
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('특성 중요도 (Random Forest)', fontsize=14)
plt.tight_layout()
plt.savefig('california_feature_importance.png')
plt.close()

# 시각화: 실제값 vs 예측값
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title('실제값 vs 예측값', fontsize=14)
plt.tight_layout()
plt.savefig('california_actual_vs_predicted.png')
plt.close()

# 시각화: 잔차 분포
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('잔차')
plt.ylabel('빈도')
plt.title('잔차 분포', fontsize=14)
plt.tight_layout()
plt.savefig('california_residuals.png')
plt.close()

# 시각화: 잔차 vs 예측값 (모델 성능 진단)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 vs 예측값', fontsize=14)
plt.tight_layout()
plt.savefig('california_residuals_vs_predicted.png')
plt.close()

print("\n----- 모델 분석 완료 -----")
print("4개의 그래프가 저장되었습니다:")
print("1. california_feature_importance.png - 특성 중요도")
print("2. california_actual_vs_predicted.png - 실제값 vs 예측값")
print("3. california_residuals.png - 잔차 분포")
print("4. california_residuals_vs_predicted.png - 잔차 vs 예측값")

----- California Housing 데이터셋 로드 중 -----
----- California Housing 데이터셋 정보 -----
특성 개수: 8
샘플 개수: 20640
타겟 변수(주택 가격) 통계:
  - 최소값: 0.15
  - 최대값: 5.00
  - 평균값: 2.07
  - 표준편차: 1.15

----- 훈련/테스트 데이터셋 크기 -----
X_train: (16512, 8)
X_test: (4128, 8)

----- 특성 미리보기 -----
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24
3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25
4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25

----- Optuna 최적화 시작 (50회 시도) -----
[I 2025-07-15 11:50:45,872] A new study created in memory with name: no-name-aaa75991-c58c-496b-88be-7a19fce4e4d6
[I 2025-07-15 11:50:49,398] Trial 0 finished with value: 0.5071440509938875 and parameters: {'max_depth': 25, 'min_samples_leaf': 4, 'min_samples_split': 7, 'n_estimators': 193, 'max_features': 0.6664474925721384}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:50:50,154] Trial 1 finished with value: 0.7188922666291981 and parameters: {'max_depth': 4, 'min_samples_leaf': 8, 'min_samples_split': 6, 'n_estimators': 91, 'max_features': 0.9801725433989807}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:50:52,147] Trial 2 finished with value: 0.6038663722549712 and parameters: {'max_depth': 7, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 241, 'max_features': 0.5447327218369955}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:50:55,878] Trial 3 finished with value: 0.5198634701682306 and parameters: {'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 245, 'max_features': 0.6578439089213309}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:50:58,021] Trial 4 finished with value: 0.5636978052428379 and parameters: {'max_depth': 9, 'min_samples_leaf': 10, 'min_samples_split': 8, 'n_estimators': 184, 'max_features': 0.6839032672846387}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:50:58,886] Trial 5 finished with value: 0.7193043675053106 and parameters: {'max_depth': 4, 'min_samples_leaf': 7, 'min_samples_split': 7, 'n_estimators': 104, 'max_features': 0.9982690031614305}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:50:59,821] Trial 6 finished with value: 0.6784771222240584 and parameters: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 179, 'max_features': 0.4157278418082858}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:51:00,901] Trial 7 finished with value: 0.6717679996097752 and parameters: {'max_depth': 5, 'min_samples_leaf': 7, 'min_samples_split': 6, 'n_estimators': 171, 'max_features': 0.5971057514834507}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:51:03,194] Trial 8 finished with value: 0.5140544971553479 and parameters: {'max_depth': 16, 'min_samples_leaf': 5, 'min_samples_split': 7, 'n_estimators': 111, 'max_features': 0.9205598903071066}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:51:04,380] Trial 9 finished with value: 0.5282448322388069 and parameters: {'max_depth': 13, 'min_samples_leaf': 9, 'min_samples_split': 9, 'n_estimators': 57, 'max_features': 0.8997504151707438}. Best is trial 0 with value: 0.5071440509938875.
[I 2025-07-15 11:51:10,292] Trial 10 finished with value: 0.5040546922545637 and parameters: {'max_depth': 25, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 291, 'max_features': 0.7598911295338786}. Best is trial 10 with value: 0.5040546922545637.
[I 2025-07-15 11:51:17,036] Trial 11 finished with value: 0.5042078907099623 and parameters: {'max_depth': 25, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 296, 'max_features': 0.7857819124275864}. Best is trial 10 with value: 0.5040546922545637.
[I 2025-07-15 12:42:44,088] Trial 12 finished with value: 0.5008563398961312 and parameters: {'max_depth': 25, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 292, 'max_features': 0.7974999894944904}. Best is trial 12 with value: 0.5008563398961312.
[I 2025-07-15 12:42:53,032] Trial 13 finished with value: 0.5021264826363405 and parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 294, 'max_features': 0.823163259129655}. Best is trial 12 with value: 0.5008563398961312.
[I 2025-07-15 12:43:01,570] Trial 14 finished with value: 0.499771836627335 and parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 259, 'max_features': 0.8225834146452452}. Best is trial 14 with value: 0.499771836627335.
[I 2025-07-15 12:43:08,970] Trial 15 finished with value: 0.49951479100130997 and parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 248, 'max_features': 0.8412097779665711}. Best is trial 15 with value: 0.49951479100130997.
[I 2025-07-15 12:43:17,737] Trial 16 finished with value: 0.4993816903585843 and parameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 241, 'max_features': 0.8676868292102558}. Best is trial 16 with value: 0.4993816903585843.
[I 2025-07-15 12:43:21,133] Trial 17 finished with value: 0.49824877734273754 and parameters: {'max_depth': 19, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 214, 'max_features': 0.35529881837281757}. Best is trial 17 with value: 0.49824877734273754.
[I 2025-07-15 12:43:23,637] Trial 18 finished with value: 0.5025562171876969 and parameters: {'max_depth': 17, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 217, 'max_features': 0.3411604253953329}. Best is trial 17 with value: 0.49824877734273754.
[I 2025-07-15 12:43:25,742] Trial 19 finished with value: 0.5027997168830305 and parameters: {'max_depth': 22, 'min_samples_leaf': 4, 'min_samples_split': 3, 'n_estimators': 155, 'max_features': 0.4800760884671358}. Best is trial 17 with value: 0.49824877734273754.
[I 2025-07-15 12:43:27,814] Trial 20 finished with value: 0.5001858910715052 and parameters: {'max_depth': 17, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 212, 'max_features': 0.30851895328148077}. Best is trial 17 with value: 0.49824877734273754.
[I 2025-07-15 12:43:36,582] Trial 21 finished with value: 0.5023307809771481 and parameters: {'max_depth': 21, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 263, 'max_features': 0.8971596188464702}. Best is trial 17 with value: 0.49824877734273754.
[I 2025-07-15 12:43:42,128] Trial 22 finished with value: 0.5025489079506278 and parameters: {'max_depth': 18, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 222, 'max_features': 0.7413058691164374}. Best is trial 17 with value: 0.49824877734273754.
[I 2025-07-15 12:43:44,695] Trial 23 finished with value: 0.5073701397658338 and parameters: {'max_depth': 15, 'min_samples_leaf': 4, 'min_samples_split': 3, 'n_estimators': 143, 'max_features': 0.5839684901116605}. Best is trial 17 with value: 0.49824877734273754.
[I 2025-07-15 12:43:49,752] Trial 24 finished with value: 0.49369990171472766 and parameters: {'max_depth': 23, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 265, 'max_features': 0.4313744205907149}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:43:52,223] Trial 25 finished with value: 0.5035921799896816 and parameters: {'max_depth': 23, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 273, 'max_features': 0.3696053629445185}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:43:54,913] Trial 26 finished with value: 0.4964526231358034 and parameters: {'max_depth': 23, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 206, 'max_features': 0.4339463266353629}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:43:57,258] Trial 27 finished with value: 0.5117191118835973 and parameters: {'max_depth': 23, 'min_samples_leaf': 6, 'min_samples_split': 4, 'n_estimators': 199, 'max_features': 0.44408865551819077}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:00,966] Trial 28 finished with value: 0.49560544902728915 and parameters: {'max_depth': 23, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 232, 'max_features': 0.5014428240531386}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:04,057] Trial 29 finished with value: 0.5029774999706986 and parameters: {'max_depth': 23, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 227, 'max_features': 0.5026410274511792}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:05,980] Trial 30 finished with value: 0.5293565930886016 and parameters: {'max_depth': 11, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 198, 'max_features': 0.40755450426911305}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:09,292] Trial 31 finished with value: 0.4947922469152794 and parameters: {'max_depth': 18, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 272, 'max_features': 0.48190211232572566}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:13,625] Trial 32 finished with value: 0.49843504051843107 and parameters: {'max_depth': 22, 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 277, 'max_features': 0.5052611224477809}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:18,039] Trial 33 finished with value: 0.49599818814048097 and parameters: {'max_depth': 24, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 272, 'max_features': 0.5557316939149559}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:22,074] Trial 34 finished with value: 0.5001188184526248 and parameters: {'max_depth': 24, 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 275, 'max_features': 0.5612026776586047}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:25,067] Trial 35 finished with value: 0.5068713647895782 and parameters: {'max_depth': 21, 'min_samples_leaf': 5, 'min_samples_split': 7, 'n_estimators': 232, 'max_features': 0.6136231303323647}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:28,697] Trial 36 finished with value: 0.5037134588278066 and parameters: {'max_depth': 24, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 263, 'max_features': 0.5310575222896624}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:31,900] Trial 37 finished with value: 0.4958312260176937 and parameters: {'max_depth': 18, 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 254, 'max_features': 0.47527277686802094}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:34,548] Trial 38 finished with value: 0.5062248218774563 and parameters: {'max_depth': 14, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 250, 'max_features': 0.46300790690843235}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:36,953] Trial 39 finished with value: 0.5102934758474149 and parameters: {'max_depth': 18, 'min_samples_leaf': 6, 'min_samples_split': 7, 'n_estimators': 238, 'max_features': 0.3952365213630665}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:40,408] Trial 40 finished with value: 0.5251431756031859 and parameters: {'max_depth': 15, 'min_samples_leaf': 10, 'min_samples_split': 6, 'n_estimators': 251, 'max_features': 0.6265486071060304}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:44,175] Trial 41 finished with value: 0.5301786490271305 and parameters: {'max_depth': 11, 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 276, 'max_features': 0.6923940925015255}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:48,443] Trial 42 finished with value: 0.49618348034803045 and parameters: {'max_depth': 22, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 283, 'max_features': 0.5330452308547228}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:51,435] Trial 43 finished with value: 0.49873128986234266 and parameters: {'max_depth': 18, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 262, 'max_features': 0.48956258700715655}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:53,886] Trial 44 finished with value: 0.5801595977186264 and parameters: {'max_depth': 8, 'min_samples_leaf': 8, 'min_samples_split': 6, 'n_estimators': 284, 'max_features': 0.5700941582480646}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:44:57,938] Trial 45 finished with value: 0.4996646180813318 and parameters: {'max_depth': 24, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 264, 'max_features': 0.5244050738026709}. Best is trial 24 with value: 0.49369990171472766.
[I 2025-07-15 12:45:01,607] Trial 46 finished with value: 0.4936072863292827 and parameters: {'max_depth': 21, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 298, 'max_features': 0.4644079515078592}. Best is trial 46 with value: 0.4936072863292827.
[I 2025-07-15 12:45:05,045] Trial 47 finished with value: 0.49797387371897556 and parameters: {'max_depth': 19, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 300, 'max_features': 0.4483482157223933}. Best is trial 46 with value: 0.4936072863292827.
[I 2025-07-15 12:45:08,841] Trial 48 finished with value: 0.4933246830834971 and parameters: {'max_depth': 21, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 285, 'max_features': 0.38571816486251764}. Best is trial 48 with value: 0.4933246830834971.
[I 2025-07-15 12:45:12,524] Trial 49 finished with value: 0.49412685134389006 and parameters: {'max_depth': 21, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 287, 'max_features': 0.38709760947249117}. Best is trial 48 with value: 0.4933246830834971.

----- 최적화 결과 -----
최소 RMSE: 0.4933
최적 하이퍼파라미터: {'max_depth': 21, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 285, 'max_features': 0.38571816486251764}

----- 최종 모델 평가 -----
RMSE: 0.4933
MAE: 0.3220
R²: 0.8143

----- 특성 중요도 -----
      feature  importance
0      MedInc    0.406333
5    AveOccup    0.126552
6    Latitude    0.121427
7   Longitude    0.117374
2    AveRooms    0.106557
1    HouseAge    0.053635
3   AveBedrms    0.038807
4  Population    0.029316
...c:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\머신러닝\0715파이프라인_캘리포니아집값.py:183: UserWarning: Glyph 52264 (\N{HANGUL SYLLABLE CA}) missing from font(s) DejaVu Sans.
  plt.savefig('california_residuals_vs_predicted.png')

----- 모델 분석 완료 -----
4개의 그래프가 저장되었습니다:
1. california_feature_importance.png - 특성 중요도
2. california_actual_vs_predicted.png - 실제값 vs 예측값
3. california_residuals.png - 잔차 분포
4. california_residuals_vs_predicted.png - 잔차 vs 예측값