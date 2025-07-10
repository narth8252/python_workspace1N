"""
특성이 아주많을때(고차원),암환자같은 30개의 특성을 갖고
이미지 → 80 by 80 by 3d = 19200개의 특성(feature)
주성분분석, 원특성으로 원특성의 특이점을 설명할수있는 새특성 생성
19200 → 10개 뽑아서 학습(정확한 정의 불가)
학습시간 감소, 단시간에 높은 학습효과
비지도학습의 일종 → 시각화
분석시, 상관계수
특성이 300개면
각성분별로 히스토그램그리거나 히트맵 활용
산포도,히스토그램,히트맵 그리기
"""
from sklearn.datasets import load_iris, load_breast_cancer
# from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = load_iris()
# iris = load_breast_cancer()
df1 = pd.DataFrame(iris['data'], \
                   columns=iris['feature_names'])
df1['target'] = iris['target']
print(df1.head())

#seaborn에서 산포도행렬(pairplot) 차트900개라 너무오래걸려서 주석막음
# sns.pairplot(df1, hue="target")
#setosa, vesicolor
# plt.show() #모든

# seaborn 에서 산포도행렬(pairplot)
# sns.pairplot(df1, hue="target")
# setosa, versicolor, virginica 세 가지 색상을 다르게 표현한다.
# plt.show() # 모든 그래픽 출력은 pyplot을 사용해야 한다

# 특성이 많을 경우에 상관관계 시각화 - 히스토그램
# 학생과 암석 두 개의 클래스를 갖는 데이터들의 집합, 두 클래스별로 각자 데이터를 모아서 히스토그램을 그려보자
# 히스토그램에 특성의 개수만큼 나온다. 히스토그램이 데이터의 분포도를 확인하기 좋은 차트다
# 구간을 나눠서 => 히스토그램을 그려보자

#데이터를 악성,양성으로 쪼개자
cancer = load_breast_cancer()
cancer_data = cancer['data']
cancer_target = cancer['target']

#악성끼리 모으자
malignant = cancer_data[cancer_target==0] #악성종양데이터만
benign = cancer_data[cancer_target==1] #양성종양
print(malignant.shape)
print(benign.shape)

import numpy as np
#히스토그램을 30개그려보자
#차트in차트를 작게 계속만든다. 15 by 2로 나눠 각각공간에 하나의차트만.
#반환값이 차트자체와 축에대한정보, 
# figsize=(10,20) 차트전체크기 width by height inch단위
fig, axes = plt.subplots(15,2, figsize=(10,20))
ax = axes.ravel() #축에대한 정보가져오기
for i in range(30): #특성(feature) 30개
    #1. 구간나누기
    count, bins = np.histogram(cancer_data[:, i], bins=50)
    #count각구간별데이터개수,    (첫열을 50개로 나눠라)
    #bins구간리스트                           
    #co
