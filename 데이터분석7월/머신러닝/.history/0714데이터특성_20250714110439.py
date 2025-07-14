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

file_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
print(f"[파일 경로 확인] {file_path}")
print(f"[파일 존재 여부] {os.path.exists(file_path)}")

data = pd.read_csv(file_path,
                   header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income'])

data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
print("[원본 데이터]")
print(data.head())

data = pd.get_dummies(data)
print("[원핫 인코딩 결과]")
print(data.head())

print("[컬럼 목록]")
print(data.columns)

print("✅ 모든 처리 완료")

#타겟까지 넣는바람에 타겟도 원핫인코딩 된상태
X = data.loc[:, 'agd':'occupation_ Transport-moving']
y = data.loc[:, 'income_ >50K']
print(X.head())
print(y.head())
