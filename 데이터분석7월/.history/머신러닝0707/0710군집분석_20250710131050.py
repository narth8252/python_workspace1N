# 0710 pm1시 
"""
# 군집(Clustering, 클러스터링)
ㆍ
ㆍ
----------------------------------------------
군집 - 클러스터링, 결과를 안준다. 그냥 데이터만 가지고 분류하는데, 대강 군집개수 알려줘야함
        np.random.normal(평균, 표준편차, 형태) - 평균과 표준편차를 만족하는 가우스분포를 따르는 데이터생성
        np.random.normal(173, 10, 10)
"""

import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

#테스트용 예제
a = np.random.normal(173, 10, 3000)
print(a[:10]) #데이터만으면 앞에10개만 잘라서 출력해서 확인
# [153.92511725 195.5680131  157.15041973 173.09840246 172.65846362
#  168.55172158 170.6194423  175.97373577 182.52604063 178.11583603]
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(a, bins=(30))
plt.show() #

# 1. 데이터 불러오기
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# X = cancer['data']
# y = cancer['target']

# # 2. 스케일링 (표준화)
# from sklearn.preprocessing import StandardScaler
# scalar = StandardScaler()
# scalar.fit(X)
# X_scaled = scalar.transform(X)

# 3. 주성분 분석 (PCA)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)  # 2개의 성분으로 차원 축소(성분개수지정)
# pca.fit(X_scaled)         #학습
# X_pca = pca.transform(X_scaled)

# # 4. 데이터를 train/test로 분할 (원본 / 스케일링 / PCA)
# from sklearn.model_selection import train_test_split
# # (4-1) 원본 데이터 그대로 
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# # (4-2) 스케일된 데이터 기준
# X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled =\
#         train_test_split(X_scaled, y, random_state=0)
# # (4-3) PCA 차원축소된 데이터 기준
# X_train_pca, X_test_pca, y_train_pca, y_test_pca = \
#         train_test_split(X_pca, y, random_state=0)
# print("===============================================================")
# from sklearn.linear_model import LogisticRegression

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


# 7. PCA 시각화
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='purple', alpha=0.5, label='Malignant')
# plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='green', alpha=0.5, label='Benign')
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.title('PCA of Breast Cancer Dataset')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

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