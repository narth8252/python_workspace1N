import pandas as pd
import numpy as np
import os #파일이나 폴더경로 지정시 필요

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# ✔ STEP 1  ── 데이터 불러오기 & 필요 없는 열 제거
# -------------------------------------------------
# path = r"C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data"
train = pd.read_csv("./data/titanic_train.csv")
test  = pd.read_csv("./data/titanic_test.csv")

#1.불필요한열삭제
print("---------1.불필요한열삭제-----------------") 
print(train.head()) #원본데이터 inplace=True안먹히는함수많아 반환값받고 shpe찍어서 확인
# DROP_COLS = ["PassengerId", "Name", "Ticket", "SibSp", "Parch"]
train=train.drop(columns=["PassengerId", "Name", "Ticket", "SibSp", "Parch", "Cabin"])
test=test.drop(columns=["PassengerId", "Name", "Ticket", "SibSp", "Parch", "Cabin"])
print(train.head())
print(train.shape) #특성4개 삭제함


#2.결측치 확인후 대체
print("---------2.결측치 확인 후 대체-----------------") 
print(train.isna().sum()) #각특성별로 NaN개수 출력
#Age나 Embarked는 대체
print(train.info())
print(train.describe()) #평균값, 중간값, 최빈값 등 뭐가 나을지 지정하기 위해 써보자

#2-1.Age는 mean=평균값으로 대체
age_mean = train["Age"].mean() #mean=평균값
train['Age'].fillna(age_mean, inplace=True) #반환값이 아닌 자기자신이 바뀜
test['Age'].fillna(age_mean, inplace=True) #반환값이 아닌 자기자신이 바뀜
print(train['Age'].isna().sum())
print(train.isna().sum())
print(test.isna().sum())

#2-2.Embarked는 데이터무의미하니 행삭제
#행중에 한컬럼이라도 NaN값있으면 전체행 삭제
train = train.dropna(axis=0, how='any')
test = test.dropna(axis=0, how='any')

# for df in (train, test):
#     df["Age"].fillna(df["Age"].mean(), inplace=True)           # 연속형 → 평균
#     df["Fare"].fillna(df["Fare"].median(), inplace=True)       # 연속형 → 중앙값
#     df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # 범주형 → 최빈값
#     df["Cabin"] = df["Cabin"].fillna("N").str[0]               # 결측 → 'N', 앞글자만

#3.이상치 제거 boxplot그리기(IQR 1.5배 바깥값)
print("---------3.이상치 제거(boxplot)-----------------")
train.boxplot
# train = remove_outliers(train, ["Age", "Fare"])

#4. 중복 행 제거
print("---------4.중복 행 제거-----------------")
# train.drop_duplicates(inplace=True)

# # -------------------------------------------------
# # ✔ STEP 5  ── 값 오류 수정 (범주형 값 이상 여부 검토)
# # -------------------------------------------------
# # Sex와 Embarked의 고유값이 예상 범주와 다르면 찍어보기
# for col in ["Sex", "Embarked", "Cabin"]:
#     bad = train[~train[col].isin(train[col].unique())]
#     if not bad.empty:
#         print(f"[WARNING] {col} 이상값:\n", bad[col].value_counts())

# # -------------------------------------------------
# # ✔ STEP 6  ── 라벨/원‑핫 인코딩
# # -------------------------------------------------
# # Cabin(앞글자), Embarked, Pclass(범주형)은 원‑핫 / Sex는 0‑1 매핑으로 처리
# CATEGORICAL = ["Cabin", "Embarked", "Pclass"]
# BINARY       = ["Sex"]
# NUMERICAL    = ["Age", "Fare"]

# # Sex를 0/1로
# train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
# test["Sex"]  = test["Sex"].map({"male": 0, "female": 1})

# # -------------------------------------------------
# # ✔ STEP 7  ── 스케일링 + 모델 파이프라인
# # -------------------------------------------------
# target = train["Survived"]
# features = train.drop("Survived", axis=1)

# X_train, X_val, y_train, y_val = train_test_split(
#     features, target, test_size=0.2, random_state=42, stratify=target
# )

# # 컬럼 변환기
# preprocess = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), NUMERICAL),
#         ("cat", "passthrough",  CATEGORICAL),   # 원‑핫은 get_dummies로 미리 하지 않고 RF라서 그대로 둬도 ok
#         ("bin", "passthrough",  BINARY)
#     ],
#     remainder="drop"
# )

# rf = RandomForestClassifier(random_state=42)

# pipe = Pipeline(steps=[
#     ("prep", preprocess),
#     ("model", rf)
# ])

# param_grid = {
#     "model__n_estimators": [50, 100, 200],
#     "model__max_depth":    [None, 5, 10],
#     "model__min_samples_split": [2, 5]
# }

# grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", verbose=1)
# grid.fit(X_train, y_train)

# # -------------------------------------------------
# # ✔ STEP 8  ── 성능 평가 & 특성 중요도
# # -------------------------------------------------
# best_pipe = grid.best_estimator_
# y_pred = best_pipe.predict(X_val)

# print("📊 최적 파라미터:", grid.best_params_)
# print("✅ 검증 정확도:", accuracy_score(y_val, y_pred))
# print("\n📄 분류 리포트:\n", classification_report(y_val, y_pred))

# # 특성 중요도 (파이프라인 내부에서 RF가 학습됐으니 가져와서 출력)
# rf_model = best_pipe.named_steps["model"]
# feature_names = (
#     NUMERICAL                                         # 스케일링된 수치
#     + CATEGORICAL                                     # 그대로 통과
#     + BINARY                                          # Sex
# )
# importances = pd.Series(rf_model.feature_importances_, index=feature_names)
# importances.sort_values(ascending=False, inplace=True)

# print("\n🔍 Feature Importances")
# print(importances.round(3))
