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
        ("onehot", OneHotEncoder(sparse_output=False), ['workclass', 'education', 'gender', 'occupation'])
    ]
)

# 학습 및 변환
ct.fit(X)
X_transformed = ct.transform(X)

# 결과 확인
print("[ColumnTransformer 적용 결과]")
print(X_transformed[:5])  # 앞 5개 샘플만 출력
print("[변환된 shape]", X_transformed.shape)

"""
[원본 데이터]
   age          workclass   education   gender  hours-per-week          occupation  income
0   39          State-gov   Bachelors     Male              40        Adm-clerical   <=50K
1   50   Self-emp-not-inc   Bachelors     Male              13     Exec-managerial   <=50K
2   38            Private     HS-grad     Male              40   Handlers-cleaners   <=50K
3   53            Private        11th     Male              40   Handlers-cleaners   <=50K
4   28            Private   Bachelors   Female              40      Prof-specialty   <=50K
[ColumnTransformer 적용 결과]
[[ 0.03067056 -0.03542945  0.          0.          0.          0.
   0.          0.          0.          1.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          1.          0.          0.          0.
   0.          0.          0.          0.          1.          0.
   1.          0.          0.          0.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.        ]
 [ 0.83710898 -2.22215312  0.          0.          0.          0.
   0.          0.          1.          0.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          1.          0.          0.          0.
   0.          0.          0.          0.          1.          0.
   0.          0.          0.          1.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.        ]
 [-0.04264203 -0.03542945  0.          0.          0.          0.
   1.          0.          0.          0.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          0.          0.          1.          0.
   0.          0.          0.          0.          1.          0.
   0.          0.          0.          0.          0.          1.
   0.          0.          0.          0.          0.          0.
   0.          0.        ]
 [ 1.05704673 -0.03542945  0.          0.          0.          0.
   1.          0.          0.          0.          0.          0.
   1.          0.          0.          0.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          0.          0.          1.          0.
   0.          0.          0.          0.          0.          1.
   0.          0.          0.          0.          0.          0.
   0.          0.        ]
 [-0.77576787 -0.03542945  0.          0.          0.          0.
   1.          0.          0.          0.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          1.          0.          0.          0.
   0.          0.          0.          1.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          0.          1.          0.          0.
   0.          0.        ]]
[변환된 shape] (32561, 44)
"""