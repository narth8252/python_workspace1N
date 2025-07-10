"""
특성이 아주 많을 때(고차원), 예: 암 환자 데이터는 30개의 특성을 가짐
이미지 데이터 → 80 x 80 x 3 = 19200개의 특성(feature)
주성분분석(PCA): 원래 특성에서 중요한 특성을 뽑아 새로운 특성 생성
예: 19200개 중 10개만 뽑아서 학습 → 정확한 정의는 어려움
→ 학습 시간 감소, 단시간에 높은 학습 효과
→ 비지도 학습의 일종으로 시각화에 유용
→ 분석 시 상관관계, 상관계수 고려
특성이 300개면:
→ 각 성분별로 히스토그램 그리기, 히트맵 활용
→ 산포도, 히스토그램, 히트맵 등 시각화 사용
"""
from sklearn.datasets import load_iris, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Iris 데이터 불러오기 (기본 예시용)
iris = load_iris()
# iris = load_breast_cancer()
df1 = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df1['target'] = iris['target']
print(df1.head())

#seaborn에서 산포도행렬(pairplot)시각화 차트900개라 너무오래걸려서 주석막음
# setosa, versicolor, virginica 세 가지 색상을 다르게 표현한다.
# plt.show() # 모든 그래픽 출력은 pyplot을 사용해야 한다

# 특성이 많을 경우에 상관관계 시각화 - 히스토그램
# 학생과 암석 두 개의 클래스를 갖는 데이터들의 집합, 두 클래스별로 각자 데이터를 모아서 히스토그램을 그려보자
# 히스토그램에 특성의 개수만큼 나온다. 히스토그램이 데이터의 분포도를 확인하기 좋은 차트다
# 구간을 나눠서 => 히스토그램을 그려보자

# 유방암 데이터 불러오기
cancer = load_breast_cancer()
cancer_data = cancer['data']
cancer_target = cancer['target']

# 악성/양성 데이터 분리
malignant = cancer_data[cancer_target == 0]  # 악성 종양 데이터
benign = cancer_data[cancer_target == 1]     # 양성 종양 데이터
print("Malignant shape:", malignant.shape)
print("Benign shape:", benign.shape)

# 히스토그램 30개 그리기 (15행 2열)
#차트in차트를 작게 계속만든다. 15 by 2로 나눠 각각공간에 하나의차트만.
#반환값이 차트자체와 축에대한정보, 
# figsize=(10,20) 차트전체크기 width by height inch단위
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
ax = axes.ravel()  # 2D 배열을 1D로 펼쳐서 인덱싱 #축에대한 정보가져오기

for i in range(30):  # 특성 30개
    # 각 특성마다 히스토그램 구간 계산
    count, bins = np.histogram(cancer_data[:, i], bins=50)
    #count각구간별데이터개수,    (첫열을 50개구간으로 나눠라)
    #bins구간리스트     
                    
    # 악성/양성 각각 히스토그램 시각화
    ax[i].hist(malignant[:, i], bins=bins, color='purple', alpha=0.5, label='malignant')
    ax[i].hist(benign[:, i], bins=bins, color='green', alpha=0.5, label='benign')
    ax[i].set_title(cancer['feature_names'][i])
    ax[i].set_yticks([])  # y축 눈금 제거

# 첫 번째 서브플롯에 범례 및 라벨 추가
ax[0].set_xlabel('Feature Value')
ax[0].set_ylabel('Frequency')
ax[0].legend(loc='best')
# ax[0].legend(['malignant','benign'], loc='best') #범수-각각이 의미하는바

# 전체 레이아웃 정리 및 출력
fig.tight_layout() #차트재정렬
plt.show()

#히트맵 산포도행렬 못그릴때 자주사용하는 차트 - 상관관계확인
#1. 상관관계 행렬만들기
correlation_matrix = df1.corr() 
#데이터프레임이 corr이라는 함수가 있어서 상관계수 계산
print(correlation_matrix[:10])

# 신기술 - plotly : R 언어에 있던 기능이 파이썬에 들어옴
# fast-api : 풀스택 개발 - 프론트랑 백엔드를 쪼개 작성. 
#            프론트엔드는 html, css, javascript가 독립.
#            백엔드는 json 형태로 데이터를 주고받는 역할만 해야 한다.
#            → 백엔드에 필요한 기술만 집약. 
#            → 장고든 플라스크든 별불요 없어지고 있다는 생각이 듦.
#            node.js : 애초에 자바스크립트라 기본적으로 json으로 주고받는 게 디폴트.

#2.히트맵그리기
annot=True #차트 줄속성, 히트맵의 셀에 값표시, False는 표시안함
cmap = 'coolwarm' #히트맵에서 가장많이 사용하는색상(양은 적색, 음의관계는 파란색으로 짙어짐)
ftm ='.2f' #표시될 숫자의 소수점자리수 지정
sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=ftm, linewidths=.5)
# sns.heatmap()은 Python에서 데이터 시각화 라이브러리 중 하나인 seaborn의 함수로, 데이터의 행과 열에 따라 색상으로 값을 표시해주는 '히트맵(heatmap)'을 생성
# correlation_matrix: 히트맵으로 표현할 2차원 데이터(예: 상관 계수 행렬)
# annot=annot: 각칸에 데이터값 표시여부 설정(True셀내부에 값표기,직접 값이나 배열을 넘길 수도 있습니다.)
# 