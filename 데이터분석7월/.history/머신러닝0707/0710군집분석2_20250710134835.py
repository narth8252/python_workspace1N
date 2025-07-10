# 0710 pm1시 
"""
# 군집(Clustering, 클러스터링)
ㆍ정의: 라벨 없는 데이터(X)만 보고 비지도학습(unsupervised learning) 방식으로, 비슷한 특성을 가진 데이터들을 그룹(클러스터) 으로 나눔.
ㆍ결과: "정답(label)"이 없기 때문에 모델이 자동으로 그룹을 찾아냄.
ㆍ입력: 데이터만 있으면 됨 (라벨 불필요). 특징 데이터 (X), 정답 (y) 없음.
ㆍ출력: 각 데이터가 어느 군집(cluster)에 속하는지 알려주는 군집 레이블.
ㆍ군집수K(n_cluster): 알고리즘에 따라 직접 지정/추정. 예: KMeans(n_clusters=3)
                    엘보우법, 실루엣 점수, 전문가 판단 등을 활용.
ㆍ활용: 고객 세분화, 문서 분류, 이상 탐지 등.
ㆍ군집 개수 설정 영향: 너무 적게 지정 → 다른 군집끼리 합쳐져 정보 손실.
                    너무 많이 지정 → 실제 의미 없는 쪼갬 발생.

# np.random.normal(mean평균, std표준편차, size형태)
ㆍ정의: 정규분포(가우시안 분포)를 따르는 랜덤 데이터를 생성.
ㆍmean: 평균 (데이터가 중심을 가질 위치)
ㆍstd: 표준편차 (데이터의 흩어짐 정도)
ㆍsize: 생성할 데이터 수 혹은 shape. (행, 열) 형태 가능.
군집은 데이터만 가지고 분류하고, np.random.normal은 정규분포 기반의 더미 데이터를 만들며, 히스토그램으로 그 분포를 시각화할 수 있다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" #숫자는 실제 물리 코어 수로 지정 (예: 4, 6 등)
os.environ["OMP_NUM_THREADS"] = "1"    #Windows+MKL KMeans 메모리누수방지

# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression



# 1. 데이터 생성 및 로딩
np.random.seed(42)
x1 = np.random.normal(0, 1, (50,2)) #형태가 tuple로 2D로
x2 = np.random.normal(5, 1, (50,2)) #형태가 tuple로 2D로
x3 = np.random.normal(2.5, 1, (50,2)) #형태가 tuple로 2D로

# 2. 데이터 합치기 concatenate-인자들이 dataframe이어야한다
#np.vstack 사용해 세로로결합
X = load_iris()['data']  # 라벨 없는 iris 데이터 (150 x 4)
print(X.shape) #y없음, y모름
print(X[:10])

#n_clusters = 3 :몇개가 묶였는지 알려줘야 효율적인 군집분석
#n_clusters 모를때는 엘보우,실루엣,전문가견해로
#원래 군집개수보다 적게주면 그나마 차이적은 군집끼리 합친다. 정보손실, 해석어려워짐
#원래 군집개수보다 많이 주면 강제로 군집을 더 쪼갠다

# 3. KMeans 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X) #X에 데이터 학습
y_kmeans = kmeans.predict(X) # 군집 예측
print(y_kmeans[:20])

# 4. 군집 중심 출력: 군집분석시 각군집의 중심값 가져온다
center = kmeans.cluster_centers_  # 각 군집의 중심 좌표
print("중심값", center)

# 5. 군집분석(클러스터링, Clustering) 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 2], X[:, 3], c =y_kmeans, cmap='viridis', s=50, alpha=0.6) # 꽃잎 길이, 폭 기준으로 색칠
plt.scatter(center[:, 2], center[:, 1], color='red', s=200, marker='X', label='centroids') # 중심좌표는 빨간 X로 표시
plt.legend()
plt.grid(True)
plt.show()

"""
1. 엘보우 방법 (Elbow Method)
ㆍ목적: 클러스터 수에 따른 **오차 제곱합(WCSS)**을 시각화하여 급격히 줄어들다 꺾이는 지점을 찾는 방법.
ㆍ지표: WCSS (Within-Cluster Sum of Squares)

2. 실루엣 점수 (Silhouette Score)
ㆍ목적: 군집 내부는 가깝고, 군집 간 거리는 멀수록 점수가 높아짐.
ㆍ지표: -1 ~ 1 사이, 1에 가까울수록 좋음.
"""
from sklearn.metrics import silhouette_score

# 1. 엘보우 방법 (WCSS)
wcss = []
silhouette_scores = []
k_range = range(2, 11)  # 2~10개의 군집을 비교

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_가 바로 WCSS
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# 2. 엘보우 그래프
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method - WCSS vs K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)

# 3. 실루엣 그래프
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='s', color='green')
plt.title('Silhouette Score vs K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()
