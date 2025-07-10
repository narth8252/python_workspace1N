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
numeric_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

#StandardScaler 객체생성 및 적용
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n--- 스케일링 및 인코딩후 데이터 샘플 ---")
print(df.head())

# 5.서포트벡터머신(SVM)모델링
#전처리가 완료된 데이터로 SVM모델 학습 및 평가
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#독립변수(X)와 종속변수(y) 분리
X = df.drop('Survived', axis=1)
y = df['Surciced']

#학습데이터와 테스트데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM모델생성 및 학습
svm_model = SVC(kernel='rbf', random_state=42) #RBF커널이 일반적으로 성능좋음
svm_model.fit(X_train, y_train)

#예측 및 평가
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"모델정확도: {accuracy: .4f}")
#제시된 파이프라인을 통해 데이터의 결측치,이상치를 효과적으로 처리하고,
#⭐스케일링과 인코딩을 통해 데이터를 정제한 후:중요한파트:각특성(feature)
#서포트벡터머신(SVM)으로 학습시킨결과
#생존자 예측모델을 구축함



