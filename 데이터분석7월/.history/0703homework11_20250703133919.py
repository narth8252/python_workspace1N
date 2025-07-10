

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