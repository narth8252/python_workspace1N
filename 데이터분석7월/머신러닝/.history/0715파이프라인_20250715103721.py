#파이프라인 전체적으로 유기적 연결라인
#회귀평가와 분류평가: 머신러닝 모델이 갖고있는 score함수사용
#회귀score: 결정계수 R^2, 분류: 정확도기준, 불균형셋에 대해서는 이 기준 적용가능

# 필요한 라이브러리 임포트
from sklearn.datasets import load_iris, load_breast_cancer  # 예제 데이터셋 불러오기 위한 함수들
from sklearn.model_selection import train_test_split  # 데이터를 학습용과 테스트용으로 분리하는 함수
from sklearn.preprocessing import StandardScaler  # 데이터 표준화(평균 0, 분산 1)를 위한 클래스
# from sklearn.compose import ColumnTransformer  # 현재 사용하지 않음 (주석 처리됨)
from sklearn.pipeline import Pipeline  # 여러 단계의 처리 과정을 순차적으로 연결하는 파이프라인
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 분류 알고리즘 (현재 사용하지 않음)
from sklearn.metrics import accuracy_score # 모델 평가를 위한 메트릭
from sklearn.metrics import classification_report  # 모델 평가를 위한 메트릭
from sklearn.svm import SVC  # 서포트 벡터 머신 분류기

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline #파이프라인
from sklearn.svm import SVC
from sklearn.metrics import classification_report #분류레포트 

# 붓꽃(iris) 데이터셋 불러오기
iris = load_iris()  # iris 데이터셋 로드
X = iris.data  # 특성 데이터 (꽃잎/꽃받침 길이와 폭)
y = iris.target  # 타겟 데이터 (꽃 종류: 0, 1, 2)

# 데이터를 학습용과 테스트용으로 분할 (8:2 비율)
# test_size=0.2는 전체 데이터의 20%를 테스트 데이터로 사용
# random_state=1234는 분할 결과를 항상 같게 유지하기 위한 시드값
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# 파이프라인 정의 - 여러 단계의 처리 과정을 순차적으로 실행
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 1단계: 데이터 스케일링 (평균 0, 분산 1로 표준화)
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale'))  # 2단계: SVM 분류기 적용
    # kernel='rbf': 방사 기저 함수 커널 사용
    # C=1.0: 규제 강도 (작을수록 더 강한 규제)
    # gamma='scale': 커널 계수 (특성 수와 분산에 따라 자동 조정)
])

# 학습 데이터로 모델 학습 (파이프라인 적용)
pipeline.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 수행
y_pred = pipeline.predict(X_test)

# 모델 성능 평가 및 결과 출력
# classification_report: 정확도, 정밀도, 재현율, F1 점수 등 다양한 평가 지표를 한번에 보여줌
print(classification_report(y_test, y_pred))