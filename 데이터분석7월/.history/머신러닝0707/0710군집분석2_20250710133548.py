# 0710 pm1시 
"""
# 군집(Clustering, 클러스터링)
ㆍ정의: 라벨이 없는 비지도학습(unsupervised learning) 방식으로, 비슷한 특성을 가진 데이터들을 그룹(클러스터) 으로 나눔.
ㆍ결과: "정답(label)"이 없기 때문에 모델이 자동으로 그룹을 찾아냄.
ㆍ입력: 데이터만 있으면 됨 (라벨 불필요).
ㆍ출력: 각 데이터가 어느 군집(cluster)에 속하는지.
ㆍ군집 수(K): 알고리즘에 따라 지정하거나 추정해야 함. 예: KMeans(n_clusters=3)
ㆍ활용: 고객 세분화, 문서 분류, 이상 탐지 등.

# np.random.normal(mean, std, size)
ㆍ정의: 정규분포(가우시안 분포)를 따르는 랜덤 데이터를 생성.
ㆍmean: 평균 (데이터가 중심을 가질 위치)
ㆍstd: 표준편차 (데이터의 흩어짐 정도)
ㆍsize: 생성할 데이터 수 혹은 shape
군집은 데이터만 가지고 분류하고, np.random.normal은 정규분포 기반의 더미 데이터를 만들며, 히스토그램으로 그 분포를 시각화할 수 있다.
----------------------------------------------
군집 - 클러스터링, 결과를 안준다. 그냥 데이터만 가지고 분류하는데, 대강 군집개수 알려줘야함
        np.random.normal(평균, 표준편차, 형태) - 평균과 표준편차를 만족하는 가우스분포를 따르는 데이터생성
        np.random.normal(173, 10, 10)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 숫자는 실제 물리 코어 수로 지정 (예: 4, 6 등)
os.environ["OMP_NUM_THREADS"] = "1"

# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression



# 1. 데이터 불러오기

p
np.random.seed(42)
x1 = np.random.normal(0, 1, (50,2)) #형태가 tuple로 2D로
x2 = np.random.normal(5, 1, (50,2)) #형태가 tuple로 2D로
x3 = np.random.normal(2.5, 1, (50,2)) #형태가 tuple로 2D로

#2. 데이터 합치기 concatenate-인자들이 dataframe이어야한다
#np.vstack 사용해 세로로결합
X = load_iris
X = np.vstack((x1,x2,x3)) #매개변수를  tuple로전달
print(X.shape) #y없음, y모름
print(X[:10])

#n_clusters = 3 :몇개가 묶였는지 알려줘야 효율적인 군집분석
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X) #X학습
y_kmeans = kmeans.predict(X)
print(y_kmeans[:20])
# [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
# KMeans가 예측한 클러스터 라벨이야. 지금은 대부분 같은 군집으로 분류된 것으로 보임 → 클러스터 수가 적거나 데이터 분산이 낮을 수 있음.

#3. 결과출력: 군집분석시 각군집의 중심값 가져온다
center = kmeans.cluster_centers_
print("중심값", center)
# 중심값 [[ 4.89896609  5.19447206]
#  [-0.14754998 -0.04167172]
#  [ 2.70342156  2.46624846]]


# 4. 군집분석(클러스터링, Clustering) 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c =y_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.scatter(center[:, 0], center[:, 1], color='red', s=200, marker='X', label='centroids')
plt.legend()
plt.grid(True)
plt.show()

