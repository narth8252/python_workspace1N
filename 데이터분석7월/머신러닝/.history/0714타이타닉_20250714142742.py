import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

#  1. 데이터 불러오기
# 사용자 지정 경로
path = r'C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data'

train_df = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
test_df = pd.read_csv(os.path.join(path, 'titanic_test.csv'))

print(train_df.shape)
print(train_df.head())

# 2. 데이터 전처리 함수

def preprocess(df):
    df = df.copy()

    # 1. Age 결측 → 평균
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # 2. Fare 결측 (테스트셋만 해당) → 중앙값
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # 3. Embarked 결측 → 최빈값
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # 4. Cabin → 결측이면 'N'으로 대체, 앞글자만 남김
    df['Cabin'] = df['Cabin'].fillna('N')
    df['Cabin'] = df['Cabin'].str[0]

    # 5. 성별 숫자화
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    # 6. Embarked 문자 → 숫자
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    # 7. 필요 없는 열 제거
    drop_cols = ['PassengerId', 'Name', 'Ticket']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

#  3. 학습용/테스트용 데이터셋 분리
train_clean = preprocess(train_df)
test_clean = preprocess(test_df)

# X, y 분리
X = train_clean.drop('Survived', axis=1)
y = train_clean['Survived']

# train/val 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 정의 및 그리드 탐색
# 랜덤포레스트 모델과 하이퍼파라미터 후보 정의
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid.fit(X_train, y_train)

# 5. 검증 및 평가
