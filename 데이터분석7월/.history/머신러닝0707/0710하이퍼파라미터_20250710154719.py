# 250714 am 
# 하이퍼파라미터(Hyperparameter)**는 머신러닝에서 모델 성능을 좌우하는 중요한 개념 중 하나야. 
# 개념 → 예시 → 튜닝 방법 → 실전 코드 순으로 아주 쉽게 정리해줄게.
"""
1. 하이퍼파라미터란?
머신러닝에서 학습 전에 설정해주는 값.
모델이 스스로 학습해서 얻는 파라미터와는 다름.
모델 성능에 큰 영향을 미치므로, 잘 조정해야 좋은 결과가 나옴.
        하이퍼파라미터            🆚               파라미터
정 의	 사용자 지정값	                          모델이 학습으로 찾는값
예 시	 K in KNN, max_depth in DecisionTree	선형 회귀의 가중치(weight), 편향(bias)
설정시점  학습전에 사람이직접설정	                학습중 자동업데이트

2. 주요 모델별 하이퍼파라미터 예시
| 알고리즘              | 주요 하이퍼파라미터                              |
| -------------------- | --------------------------------------------- |
| **KNN**              | `n_neighbors` (이웃 수), `weights`, `metric`   |
| **Decision Tree**    | `max_depth`, `min_samples_split`, `criterion` |
| **RandomForest**     | `n_estimators`, `max_features`, `max_depth`   |
| **SVM**              | `C`, `kernel`, `gamma`                        |
| **KMeans**           | `n_clusters`, `init`, `max_iter`              |
| **GradientBoosting** | `learning_rate`, `n_estimators`, `subsample`  |

3. 하이퍼파라미터 튜닝 방법
① 수동 조정
사람이 직접 여러 값 실험하면서 성능 비교.
② 그리드 서치 (GridSearchCV)
지정한 여러 하이퍼파라미터 조합을 전수 조사해서 가장 좋은 성능 찾음.
from sklearn.model_selection import GridSearchCV
③ 랜덤 서치 (RandomizedSearchCV)
랜덤하게 조합 샘플링 → 속도는 빠르고 성능도 괜찮음.

4. 하이퍼파라미터가 왜 중요한가?
잘못 설정하면 과적합(overfitting) 혹은 과소적합(underfitting) 발생.
튜닝은 모델 성능 향상뿐 아니라 일반화 능력 향상에 핵심.
"""
# GridSearchCV로 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(random_state=42)

# 하이퍼파라미터 후보 설정
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 4, 6],
    'max_features': ['sqrt', 'log2']
}

# 그리드 서치 객체 생성
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# 학습
grid_search.fit(X, y)

print("최적 하이퍼파라미터:", grid_search.best_params_)
print("최고 정확도:", grid_search.best_score_)
