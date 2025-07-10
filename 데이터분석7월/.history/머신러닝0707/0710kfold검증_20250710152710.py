#250710 pm3시
"""
# k-겹 교차검증 (k-fold cross-validation)
# 머신러닝에서 모델의 일반화 성능을 평가하는 대표적인 방법
ㆍ일반화 오류를 추정하기 위해 사용되는 통계적 방법
ㆍ데이터를 훈련/테스트 세트로 한 번만 나누는 것보다 더 신뢰성 높은 모델 평가.

1. 방법
ㆍ데이터를 K개의 부분집합(fold)으로 나눠서, 1개는 테스트용(validation set), 나머지 K-1개는 학습용(training set)으로 사용.
ㆍ이과정을 K번 반복해서, 모 데이터가 한번씩 테스트셋(k개의 성능평 결과)이 되게 함.
ㆍ학습된 모델을 검증 세트로 평가하여 성능 지표(예: 정확도, 정밀도)를 기록합니다.
ㆍ마지막에 K개의 성능평균을 구해 최종평가로 사용.
2. 예시
ㆍ예를 들어, 5-겹 교차 검증(5-fold cross-validation)의 경우 데이터를 5개의 폴드로 나눕니다. 
ㆍ첫 번째 반복에서는 첫 번째 폴드를 검증 세트로 사용하고 나머지 4개 폴드를 훈련 세트로 사용합니다. 
ㆍ두 번째 반복에서는 두 번째 폴드를 검증 세트로, 나머지 4개 폴드를 훈련 세트로 사용하며, 
ㆍ이 과정을 다섯 번째 폴드까지 반복합니다. 
ㆍ최종적으로 5개의 성능 평가 점수를 얻게 되며, 
ㆍ이들의 평균을 모델의 최종 성능으로 간주합니다.

3. 장점
ㆍ과적합(overfitting)방지:모델이 훈련데이터에만 과도한 최적화 방지, 새 데이터에 일반화성능을 더 잘 측정가능.
ㆍ단일 train/test 분할보다 안정적인 성능 추정가능.
ㆍ안정적인 성능평가: 데이터를 한번만 나누는 것보다 모델의 성능을 안정적이고 신뢰성있게 평가(모든데이터가 최소 한번은 검증에 사용)
ㆍ데이터의 효율적사용: 특히 데이터가 적을때, 모든 데이터를 훈련+검증에 활용가능해 효율적.

3. 단점
ㆍ계산비용 증가: 모델을 k번 훈련+평가해야 하므로, 한번하는것보다 계산비용+시간이 더소요. 
                k값이 커질수록 단점부각
ㆍk값의 선택
k값은 사용자가 직접 선택해야 하는 하이퍼파라미터입니다.(일반적으로 5 또는 10)
너무 작으면 (k=2 또는 3), 훈련세트의 크기가 너무 작아져 모델이 제대로 학습되지 않을 수 있으며, 평가 결과의 분산이 커져 신뢰도가 떨어질 수 있습니다.
너무 크면 (예: 데이터 포인트의 수와 같게, Leave-One-Out Cross-Validation), 
훈련세트가 전 데이터와 거의 유사해져 평가결과의 편향은 줄어들지만, 
계산비용이 매우 커지고 각 훈련 세트가 거의 동일하여 평가 결과의 분산이 커질 수 있습니다.
따라서 k값은 계산비용과 평가신뢰도 사이의 균형을 맞추어 적절하게 선택하는 것이 중요

5. KFold vs StratifiedKFold?
항목               KFold	              StratifiedKFold
클래스불균형고려	❌ 무시	               ✅ 클래스 비율을 동일하게 유지함
사용 추천 상황	    클래스가 균형잡힌 데이터	분류 문제에서 클래스 불균형이 있을 때
ㆍKFold는 데이터를 나눠가며 학습과 테스트를 여러 번 수행 → 평균 성능 추정
ㆍKFold, cross_val_score, StratifiedKFold 등 도구 다양
ㆍ분류 문제면 보통 StratifiedKFold가 더 안정적임
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedGroupKFold #개선버전
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

data = load_iris() #Bunch 라는 클래스 타입 
print(data.keys())


print("타겟이름 ", data['target_names'])
print("파일명 ", data['filename'])
print("데이터설명")
print(data["DESCR"])

#2.데이터를 나누기: 한번저장해서 쓰는 딕셔너리 방식 (단점:매번 data[...]로 꺼내야 함)
X = data['data']   #ndarray 2차원배열
y = data['target'] #ndarray 1차원배열 
#2. 데이터 로딩: 속성 직접접근: 더 간단,직관적 (단점:구조를 파악하기엔 어려움)
# X = load_iris().data
# y = load_iris().target

# 모델 선언
model = LogisticRegression(max_iter=1000)

# KFold 설정 (5등분)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 결과 저장용 리스트
scores = []

# KFold 반복
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)

# 결과 출력
print("각 fold 정확도:", np.round(scores, 4))
print("평균 정확도:", np.round(np.mean(scores), 4))