

data = pd.read_csv('./data/iris.csv')
#NaN값 있는지 체크하기-각필드별로 NaN값 있는 개수 출력
print(data.isnull().sim())

sepal_length_mean = data['sepal.length'].mean()
sepal_width_mean = data['sepal.length'].mean()
petal_length_mean = data['sepal.length'].mean()
petal_width_mean = data['sepal.length'].mean()

data['sepal.length'].fillna( sepal_length_mean, inplace=True)
data['sepal.width'].fillna( sepal_width_mean, inplace=True)
data['petal.length'].fillna( petal_length_mean, inplace=True)
data['petal.width'].fillna( petal_width_mean, inplace=True)

#정규화함수
def normalize(columnname):
    max = data[columnname].max()
    min = data[columnname].min()
    return (data[columnname]-min)/(max-min)

data['sepal.length'] = normalize('sepal.length')
data['sepal.width'] = normalize('sepal.width')
data['petal.length'] = normalize('petal.length')

count, bins = np.histogram( data['petal.length'], bins=3)
bin_name=["A", "B", "C"]
data['petal_grade'] = pd.cur(x = data['petal.length'], bins=bins,
                             labels=bin_name,
                             include_lowest=True)
print(data)

#카테고리타입분석,분류분석,텍스트분석
#사이킷런 핏 학습하다. 이타원배열2d형태로만 입력받음
#새로운축을 추가해서 1d->2d
Y_class = np.array(data['petal_grade']).reshape(-1,1)