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

"""