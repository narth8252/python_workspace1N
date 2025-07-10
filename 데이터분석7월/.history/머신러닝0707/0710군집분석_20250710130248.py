# 0710 pm1시 
"""
# 
ㆍ성 문제가 있을 때 PCA를 통해 성능 향상, 시각화, 노이즈 제거, 과적합 방지 등 다양한 효과를 기대할 수 있습니다.
ㆍ
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
from sklearn.metrics import accuracy_score

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

# 4. 데이터를 train/test로 분할 (원본 / 스케일링 / PCA)
from sklearn.model_selection import train_test_split
# (4-1) 원본 데이터 그대로 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# (4-2) 스케일된 데이터 기준
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled =\
        train_test_split(X_scaled, y, random_state=0)
# (4-3) PCA 차원축소된 데이터 기준
X_train_pca, X_test_pca, y_train_pca, y_test_pca = \
        train_test_split(X_pca, y, random_state=0)
print("===============================================================")
from sklearn.linear_model import LogisticRegression

# # 5. 로지스틱 회귀 모델 생성 및 학습
# model = LogisticRegression()
# # (5-1) 원본 데이터 학습
# model.fit(X_train, y_train)
# print("-------- 기본 원본 데이터 학습 ---------")
# print("훈련셋 정확도:", model.score(X_train, y_train))
# print("테스트셋 정확도:", model.score(X_test, y_test))
# # (5-2) 스케일링된 데이터 학습
# model.fit(X_train_scaled, y_train)
# print("-------- 스케일링된 데이터 학습 ---------")
# print("훈련셋 정확도:", model.score(X_train_scaled, y_train))
# print("테스트셋 정확도:", model.score(X_test_scaled, y_test))
# # (5-3) PCA 적용된 데이터 학습
# model.fit(X_train_pca, y_train)
# print("-------- PCA 적용된 데이터 학습 ---------")
# print("훈련셋 정확도:", model.score(X_train_pca, y_train))
# print("테스트셋 정확도:", model.score(X_test_pca, y_test))

# 로지스틱 회귀 모델 생성 및 학습 이후에 들어가야 해.
# 즉, 기존에 model.fit() 등 여러 줄로 나눠서 모델을 세 번 학습하던 걸
# → evaluate_model() 함수 하나로 대체하면서 중복된 학습 코드도 정리해주는 거야.
# evaluate_model() 하나로 학습 + 예측 + 평가 + 시각화가 모두 끝나니까

print("-------------------------------------------------------")
# 6-1. 로지스틱 회귀 모델 생성 및 평가 함수 정의
from sklearn.metrics import (
    accuracy_score,        #정확도계산
    precision_score,       #정밀도(positive예측정확도)
    recall_score,          #재현율(Positive 잡아내는 비율)
    f1_score,              #정밀도&재현율의 조화 평균
    confusion_matrix,      #실제/예측 분류결과 행렬화
    ConfusionMatrixDisplay #confusion maatrix 시각화
) #scikit-learn(또는 sklearn) 라이브러리의 일부

def evaluate_model(name, X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"------ {name} ------")
    print("정확도  :", accuracy_score(y_test, y_pred))
    print("정밀도  :", precision_score(y_test, y_pred))
    print("재현율  :", recall_score(y_test, y_pred))
    print("F1 점수 :", f1_score(y_test, y_pred))
    print()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
    disp.plot(cmap='Purples')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
print("===============================================================")
# 6-2. 세 가지 데이터셋에 대해 평가 실행 (아래5.로지스틱 일일히 쓰는것 대신 사용가능)
evaluate_model("원본", X_train, X_test, y_train, y_test)
evaluate_model("스케일링", X_train_scaled, X_test_scaled, y_train, y_test)
evaluate_model("PCA (2D)", X_train_pca, X_test_pca, y_train, y_test)



# 7. PCA 시각화
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

# -------- 기본 원본 데이터 학습 ---------
# 훈련셋 정확도: 0.9460093896713615
# 테스트셋 정확도: 0.9440559440559441
# -------- 스케일링된 데이터 학습 ---------
# 훈련셋 정확도: 0.9906103286384976
# 테스트셋 정확도: 0.965034965034965
# -------- PCA 적용된 데이터 학습 ---------
# 훈련셋 정확도: 0.9624413145539906
# 테스트셋 정확도: 0.951048951048951
# ===============================================================
# ------ 원본 ------
# 정확도  : 0.951048951048951
# 정밀도  : 0.9882352941176471
# 재현율  : 0.9333333333333333
# F1 점수 : 0.96