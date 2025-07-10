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
df.drop('Cabin', axis=1, inplace=True)

# Embarked: 탑승지는 최빈값(가장많은)으로 채우기
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\n--- 결측치 처리후 ---")
print(df.isnull().sum())

# 3.이상치처리(Outlier Detection)
#주로 수치형데이터인 Age와 Fare에서 이상치 확인후 처리, IQR(사분위범위)사용
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 - 1.5 * IQR

    #이상치를 경계값으로 대체
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] > lower_bound, lower_bound, df[column])
    return df

df = handle_outliers(df, 'Age')
df = handle_outliers(df, 'Fare')

print("\n--- 이상치 처리후 데이터요악 ---")
print(df[['Age', 'Fare']].describe())

# 4.데이터 스케일링 및 인코딩
#문자열 데이터를 순자로 변환(인코딩)후, 서포트벡터머신모델 성능향상을 위해 수치 데이처의 단위를 맞추는 스케일링 진행
# 인코딩: Sex, Ebarked 컬럼을 원핫 인코딩으로 변환
# 스케일링: StandardScaler 사용하여 수치형 데이터(Age, Fare, Pclass등) 표준화
from sklearn.preprocessing import StandardScaler

#원핫 인코딩
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

#스케일링할 컬럼선택
numer




