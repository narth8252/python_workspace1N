#다이아몬드
# 다이아몬드 데이터셋 소개
# diamonds 데이터는 보석(다이아몬드)의 특징과 가격이 담긴 유명한 데이터셋이야.
# 보통 carat(캐럿), cut(컷), color(색상), clarity(투명도), depth, table 등 다양한 특성과 가격(price) 정보가 있어.

# 1단계: 데이터 불러오기
# seaborn 라이브러리에 내장되어 있어서 쉽게 불러올 수 있어.
import pandas as pd
import seaborn as sns

# 데이터 불러오기
diamonds = sns.load_dataset("diamonds")
print(diamonds.head())

# 2단계: 주요 특성 살펴보기
print(diamonds.info())
print(diamonds.describe())
print(diamonds['cut'].value_counts())

# 3단계: 전처리 (범주형 → 원핫인코딩, 수치형 스케일링)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = diamonds.drop(columns=['price'])
y = diamonds['price']

# 범주형과 수치형 나누기
categorical_cols = ['cut', 'color', 'clarity']
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(sparse_output=False), categorical_cols),
    ("scale", StandardScaler(), numerical_cols)
])

X_processed = preprocessor.fit_transform(X)
print(X_processed.shape)

# 4단계: 간단한 회귀 모델 학습