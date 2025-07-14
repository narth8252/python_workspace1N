#columnTransformer라는 클래스가 있다
#컬럼단위로 전처리작업을 해야할때  쭈욱 지정해놓으면 이것저것 적용해준다.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import mglearn
import os

file_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")

# 데이터 불러오기
data = pd.read_csv(file_path,
                   header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income'])

# 사용할 컬럼만 추림
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]

print("[원본 데이터]")
print(data.head())

# X = 입력값, y = 타겟
X = data.drop(columns='income')
y = data['income']

# ColumnTransformer 설정
ct = ColumnTransformer(
    transformers=[
        ("scaling", StandardScaler(), ['age', 'hours-per-week']),
        ("onehot", OneHotEncoder(sparse=False), ['workclass', 'education', 'gender', 'occupation'])
    ]
)

# 학습 및 변환
ct.fit(X)
X_transformed = ct.transform(X)

# 결과 확인
print("[ColumnTransformer 적용 결과]")
print(X_transformed[:5])  # 앞 5개 샘플만 출력
print("[변환된 shape]", X_transformed.shape)
