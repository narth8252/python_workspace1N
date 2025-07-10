#250709 pm4시 train_and_test2.csv
#C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\머신러닝0707
#타이타닉데이터(결측치, 이상치처리, 스케일링, 서포트벡신)

import pandas as pd
import numpy as np

#1.데이터 불러오기 및 기본탐색
df = pd.read_csv('train_and_test2.csv')
#데이터 기본정보 확인
print("--- 데이터 정보 ---")
print(df.info())

#불필요한 컬럼제거(승객ID)
df = df.drop('PassengerId', axis=1)
#결측치(Missing Values): Age, Cabin, Fare컬럼에 존재
#데이터타입: Sex, Embarked 등 문자열(object)데이터를 숫자형으로 변환

# 2.결측치처리
# Age,Fare :나이,요금 분포 왜곡되지 않게 중앙값(median)으로 채우고
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Cabin: 결측치가 너무많아(약77%) 컬럼자체 제거하는것이 합리적
df.drop('Cabin')
# Embarked: 탑승지는 최빈값(가장많은)으로 채우기





