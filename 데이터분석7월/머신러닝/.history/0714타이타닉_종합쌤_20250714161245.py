import pandas as pd
import numpy as np
import os #파일이나 폴더경로 지정시 필요
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# ✔ STEP 1  ── 데이터 불러오기 & 필요 없는 열 제거
# -------------------------------------------------
path = r"C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data"
train = pd.read_csv(os.path.join(path, "titanic_train.csv"))
test  = pd.read_csv(os.path.join(path, "titanic_test.csv"))
# train = pd.read_csv("./data/titanic_train.csv")
# test  = pd.read_csv("./data/titanic_test.csv")

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

#3.이상치 제거 boxplot그리기(IQR 1.5배 바깥값)
print("---------3.이상치 제거(boxplot)-----------------")
train.boxplot() #데이터프레임이 내부적으로 몇개의 차트 갖고있음
plt.show() #이상치확인위해 boxplot그리기

import numpy as np 
def outfliers_iqr(data):
    q1,q3 = np.percentile(data,[25,75]) #percentile은 값 2개를 넘겨받을수있다
    iqr = q3-q1
    lower_bound = q1 -(iqr*1.5)
    upper_bound = q3 +(iqr*1.5)
    return lower_bound,upper_bound # tuple형태로 두값을 반환

#2개의 필드(Fare, Age)에서 이상치 발견
for i in ['Fare', 'Age']:
    lower, upper = outfliers_iqr(train[i])
    train[i][train[i]<lower] = lower
    train[i][train[i]>upper] = upper

for i in ['Fare', 'Age']:
    lower, upper = outfliers_iqr(test[i])
    test[i][test[i]<lower] = lower
    test[i][test[i]>upper] = upper

#이상치제거됐는지 확인후 주석막기
# train.boxplot() #데이터프레임이 내부적으로 몇개의 차트 갖고있음
# plt.show() #이상치확인위해 boxplot그리기

#4.원핫인코딩
print("---------4.원핫인코딩-----------------")
train = pd.get_dummies(train)
print(train.head())
print(train.columns)

print("---------5.산포도행렬(pairplot) 시각화-----------------")
#산포도행렬이나 상관계수라도 너무 오래걸려서 포기
print(train.corr()) #상관계수 구할수없는 필드있어서 출력안됨
# import seaborn as sns
# sns.pairplot(train, diag_kind='kde',
#              hue='Survived', palette='bright')
# plt.show()

#Survived가 젤앞에 있음
X = train.iloc[:, 1:]
y = train.iloc[:, 0]
print(X.shape)
print(y.shape)

from sklearn.ensemble import Ran
model = LogisticRegression()
model.fit(X, y)
print(model.score)



