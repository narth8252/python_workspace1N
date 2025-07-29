#0714 AM10:30
"""
범주형자료의 경우 어떻게 처리할것인가? 범주형자료가 문자열로 들어오는경우도 있고
숫자형형태인경우(1대 2중 3소) 1 2 3 라벨링
범주형대이터를 정확히 찾아서 범주형으로 바꿔주고 라벨링이나 원핫인코딩
1 대
2 중
3 소

직업분류 1.학생 2.주부 3.직장인 4.프리랜서 5.회계사 6.변호사 7.교사 8.교수 ....16종
1보다 16이 큰값이라 16이 중요한값으로 인식하니 아래처럼 특성을 늘려야함
직업1 직업2 직업3 .... 직업16
1     0 0 0 0 0 0 0 0 0 0
결과는 문자열도 알아서 처리하고 있어서 굳이 다른작업 불필요
입력데이터는 반드시 작업필요
"""
# 머신러닝 모델에 넣기 위해 범주형 데이터를 원-핫 인코딩하는 전형적인 전처리 흐름
import pandas as pd
import mglearn
import os
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로 확인
file_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
print(f"[파일 경로 확인] {file_path}")
print(f"[파일 존재 여부] {os.path.exists(file_path)}")

# 데이터 불러오기
data = pd.read_csv(file_path,
                   header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income'])
print(data.head())
# 사용할 컬럼만 추림
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
print("[원본 데이터]")
print(data.head())

# 원-핫 인코딩
data = pd.get_dummies(data)
print("[원핫 인코딩 결과]")
print(data.head())
print("[컬럼 목록]")
print(data.columns)

print("✅ 모든 처리 완료")

# 입력(X), 타겟(y) 분리
# income_ >50K, income_ <=50K 중에서 >50K를 타겟으로 사용
X = data.drop(columns=['income_ >50K', 'income_ <=50K'])
y = data['income_ >50K']

# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)  # ← 경고 방지 위해 반복횟수 충분히 줌
model.fit(X, y)
print("정확도:", model.score(X, y))
