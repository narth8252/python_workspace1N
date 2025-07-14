#columnTransformer라는 클래스가 있다
#컬럼단위로 전처리작업을 해야할때  쭈욱 지정해놓으면 이것저것 적용해준다.

# 데이터 불러오기
data = pd.read_csv(file_path,
                   header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                          'income'])
print(data.head())
# 사용할 컬럼만 추림
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
print("[원본 데이터]")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

st = ColumnTransformer([
    ("scaling", StandardScaler(), 'age')
])
