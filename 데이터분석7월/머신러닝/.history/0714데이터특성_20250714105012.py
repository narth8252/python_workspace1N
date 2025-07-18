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
import pandas as pd
import mglearn
import os
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "adult.data"),
                                header=None, index_col=False,
                                names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])

data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation',
           'income']] #마지막필드가 타겟임
print(data.head())