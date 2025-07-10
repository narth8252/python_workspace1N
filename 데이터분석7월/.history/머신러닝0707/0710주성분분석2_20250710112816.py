# 0710 am11시 
"""
# PCA(Principal Component Analysis, 주성분분석) 차원 축소 정리
1. 개념 및 필요성
ㆍPCA는 특성(Feature)이 많은 데이터에서, 원래의 특성들로부터 새로운 특성(주성분)을 생성하여 차원을 축소하는 방법입니다.
ㆍ차원 축소는 데이터의 복잡성을 줄이고, 데이터 시각화 및 모델 성능 개선에 도움을 줍니다.
2. 다중공선성과 회귀분석과의 관계
ㆍ다중공선성은 여러 특성(X)들 사이에 상관관계(연관성)가 높을 때 발생합니다. 예를 들어, 유사한 댐의 수위 데이터(A1, A2, B1, C1 등)를 모두 예측에 사용할 때 다중공선성 문제가 생길 수 있습니다.
ㆍ회귀분석
 ৹ 단일선형회귀: 입력특성(X)이 1개이고 target(y)와의 관계만 고려.
 ৹ 다중선형회귀: 여러 입력특성(X1, X2, …, Xn) 사용. 이때 특성들 사이에 상호연관성이 있으면 해석이나 예측에 부정적 영향을 미칠 수 있습니다.
3. PCA의 주요 역할
ㆍ여러 특성들 사이의 상관관계를 고려하여, 서로 독립적인(상관관계가 낮은) 새로운 특성(주성분)을 생성함.
ㆍ각 주성분은 원래 특성들의 선형 결합으로 구성됩니다.
ㆍ분산이 최대가 되는 방향으로 데이터를 재구성하여, 정보의 손실을 최소화하면서 차원을 축소합니다.
ㆍ사용자가 축소할 특성(차원) 개수를 지정할 수 있습니다. 예: 30개 특성 → 2D(2개) 또는 3D(3개)로 압축
4. 장점 및 효과
ㆍ과적합 방지: 불필요한 특성(노이즈 포함)을 제거해 일반화 성능을 높임.
ㆍ계산 효율 증가: 특성 수가 줄어듦에 따라 계산 속도가 빨라짐.
ㆍ시각화: 2차원, 3차원으로 축소하면 데이터의 시각화가 쉬워짐.
ㆍ노이즈 제거: 주성분을 통해 데이터의 주요 구조만 보존, 잡음(노이즈)을 제거하는 효과.
5. PCA 알고리즘 사용 방법
ㆍfit(): 데이터의 주성분(분산이 큰 방향 벡터)들을 학습함.
ㆍtransform(): 학습된 주성분에 따라 원본 데이터를 새로운 특성(차원)으로 변환함.
6. 결론
ㆍ특성이 많거나 다중공선성 문제가 있을 때 PCA를 통해 성능 향상, 시각화, 노이즈 제거, 과적합 방지 등 다양한 효과를 기대할 수 있습니다.
ㆍfit, transform 과정을 거쳐 주성분 기반의 데이터를 얻어 학습에 활용합니다.
----------------------------------------------
특성(feature) 많을때 특성들로부터 new특성 생성
다중공선성: 댐이 여러개 있다면 A1, A2, B1, C1 .. target수량에 ㄷ
단일선형회귀: X가하나, y한테 영향미치는요소가 X하나임. 기울기,절편구하면
다중선형회귀: X들간 상호연관관계있어서 제거권장되는게 있음
            PCA가 이런부분 알아서 new요소 생성
            각각분산뽑아서 분산이 서로 최대가 되는 방향으로 회전시키고
            여러가지 조작해서 new특성 생성
            전체 특성을 재배열해서 new특성 생성- 특성개수지정가능
30 → 2D 축소, 3D이상은 시각화어려운데 시각화에 용이
    →  특성이 많으면 과적합을 깔고감 → 특성개수를 줄임으로써 과적합을 방지 → 일반화에 도움 → 계산속도 개선
  노이즈(잡음)제거
    fit, transform → PCA뽑아내고 PCA자료로 재학습.
"""

import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# 1. 데이터 불러오기
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = cancer['data']
y = cancer['target']

# 2. 스케일링 (표준화)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(X)
X_scaled = scalar.transform(X)

# 3. 주성분 분석 (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  # 2개의 성분으로 차원 축소(성분개수지정)
pca.fit(X_scaled)         #학습
X_pca = pca.transform(X_scaled)

# 4. 데이터 분할 (원본 / 스케일링 / PCA 각각)
from sklearn.model_selection import train_test_split
# 4-1. 원본 데이터를 그대로 train/test로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 2. 스케일된 데이터 기준으로 train/test 분할
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled =\
        train_test_split(X_scaled, y, random_state=0)
# 3. PCA 차원 축소된 데이터로 train/test 분할
X_train_pca, X_test_pca, y_train_pca, y_test_pca = \
        train_test_split(X_pca, y, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 로지스틱 회귀 모델 생성
model = LogisticRegression()

# 1. 원본 데이터로 학습
model.fit(X_train, y_train)
print("-------- 기본 ---------")
print("훈련셋:", model.score(X_train, y_train))
print("테스트셋:", model.score(X_test, y_test))

# 2. 스케일링 데이터로 학습
model.fit(X_train_scaled, y_train)
print("-------- 스케일링 ---------")
print("훈련셋:", model.score(X_train_scaled, y_train))
print("테스트셋:", model.score(X_test_scaled, y_test))

# 3. PCA 적용된 데이터로 학습
model.fit(X_train_pca, y_train)
print("-------- PCA ---------")
print("훈련셋:", model.score(X_train_pca, y_train))
print("테스트셋:", model.score(X_test_pca, y_test))

# 4. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='purple', alpha=0.5, label='Malignant')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='green', alpha=0.5, label='Benign')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Breast Cancer Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

