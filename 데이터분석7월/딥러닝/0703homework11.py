import pandas as pd
import numpy as np

# 1. 데이터 불러오기 및 결측치 처리
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\data 폴더에 있음.
data = pd.read_csv('./data/iris.csv')
#NaN값 있는지 체크하기-각필드별로 NaN값 있는 개수 출력
print(data.isnull().sum())
print("--------------------------")
# iris.csv 파일을 읽어와 데이터프레임으로 저장합니다.
# 각 컬럼별로 결측치(NaN) 개수를 출력합니다.

# 2. 결측치 평균값으로 대체
# 각 컬럼의 평균값을 구해 결측치를 해당 평균값으로 채웁니다.
# (참고: 모든 평균이 sepal.length로 되어 있는데, 각각의 컬럼 평균을 써야 더 정확합니다.)
sepal_length_mean = data['sepal.length'].mean()
sepal_width_mean = data['sepal.length'].mean()
petal_length_mean = data['sepal.length'].mean()
petal_width_mean = data['sepal.length'].mean()

data['sepal.length'].fillna( sepal_length_mean, inplace=True)
data['sepal.width'].fillna( sepal_width_mean, inplace=True)
data['petal.length'].fillna( petal_length_mean, inplace=True)
data['petal.width'].fillna( petal_width_mean, inplace=True)

# 3. 정규화함수: 각 컬럼을 0~1 사이 값으로 정규화합니다.
def normalize(columnname):
    max = data[columnname].max()
    min = data[columnname].min()
    return (data[columnname]-min)/(max-min)

data['sepal.length'] = normalize('sepal.length')
data['sepal.width'] = normalize('sepal.width')
data['petal.length'] = normalize('petal.length')

# 4. 구간 나누기 (binning)
# petal.length 값을 3개의 구간으로 나누고, 각 구간에 "A", "B", "C" 등급을 부여합니다.
# 결과를 petal_grade 컬럼에 저장합니다.
count, bins = np.histogram( data['petal.length'], bins=3)
bin_name=["A", "B", "C"]
import pandas as pd
data['petal_grade'] = pd.cut(x = data['petal.length'], bins=bins,
                             labels=bin_name,
                             include_lowest=True)
print(data)
print("--------------------------")

# 5. 원-핫 인코딩 (One-Hot Encoding)
#카테고리타입분석,분류분석,텍스트분석
#사이킷런 핏 학습하다. 이타원배열2d형태로만 입력받음
#새로운축을 추가해서 1d->2d
# petal_grade 등급을 2차원 배열로 변환 후, 원-핫 인코딩합니다.
# 예: "A" → [1,0,0], "B" → [0,1,0], "C" → [0,0,1]
Y_class = np.array(data['petal_grade']).reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(Y_class)

Y_class_onehot = enc.transform(Y_class).toarray()

# 6. 인코딩 복원 (원-핫 → 인덱스)
# 원-핫 인코딩된 값을 다시 등급 인덱스(0,1,2)로 복원합니다.
# 처음 10개 결과를 출력합니다.
Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
print("--------------------------")
print(Y_class_onehot[:10])
print("--------------------------")
print(Y_class_recovery[:10])

# 결측치 처리 → 정규화 → 구간 나누기 → 원-핫 인코딩 → 복원
# 데이터 전처리와 분류(카테고리) 데이터 인코딩의 기본 흐름을 연습하는 코드입니다.