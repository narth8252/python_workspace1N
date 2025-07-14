import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#  1. 데이터 불러오기
# 사용자 지정 경로
# path = r'C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data'
path = r'.\데이터분석7월\data'
train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
test = pd.read_csv(os.path.join(path, 'titanic_test.csv'))

DROP_COLS = ["PassengerID", "Name", "Ticket", "SibSp", "Parch"]
train.drop(columns=DROP_COLS, inplace=True)
test.drop(columns=[c for c in DROP_COLS if c in test.columns], 
print(train.shape)
print(train.head())

# 2. 데이터 전처리 함수
#1) PassengerID, Name, SibSp, Parch 필요없으니까 지우기
#2)각 필드별 결측치 확인
#결측치를 열을 제거하거나 행을 제거할 수도 있다.
#혹은 지나치게 결측치가 많을 경우 대체값(평균, 중간(비범주형일때는 평균 또는 중간값), 최빈값(데이터가 범주형일때))
#3)이상치제거
#4)중복값제거
#5)데이터자체가 잘못들어온값
#  value_counts함수나 Unique로 체크하기
#  값바꾸기를 시도하거나 행삭제
#6)라벨링 또는 원핫인코딩
#7)스케일링
#8)학습하고 특성의 개수가 많을경우는 특성의 중요도 확인
#   (DecisionTree 많이 사용)
#9)주성분분석
#10)여러모델로 학습하기, GridSearchCV사용도 가능

def preprocess(df):
    df = df.copy()

    # 1. Age 결측 → 평균
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    # 2. Fare 결측 (테스트셋만 해당) → 중앙값
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    # 3. Embarked 결측 → 최빈값
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # 4. Cabin → 결측이면 'N'으로 대체후 앞글자만 추출
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
train_clean = preprocess(train)
test_clean = preprocess(test)

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
best_model = grid.best_estimator_

y_pred = best_model.predict(X_val)

print("▶️ 최적 하이퍼파라미터:", grid.best_params_)
print("✅ 검증 정확도:", accuracy_score(y_val, y_pred))
print("\n📄 분류 리포트:\n", classification_report(y_val, y_pred))

# 6. 테스트 데이터 예측 (선택)
# test에는 'Survived' 없으므로 그대로 예측만 진행
test_pred = best_model.predict(test_clean)

# PassengerId가 필요하면 원본에서 따로 추출
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})

# 저장할 경우
submission.to_csv(os.path.join(path, 'submission_rf.csv'), index=False)
