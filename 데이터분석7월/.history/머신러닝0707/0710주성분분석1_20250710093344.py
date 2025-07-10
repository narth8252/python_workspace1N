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

"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df1 = pd.DataFrame(iris['data'], \
                   columns=iris['feature_names'])
df1['target'] = iris['target']
print(df1.head())
