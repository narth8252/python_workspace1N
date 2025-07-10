# C:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)\11차시_백현숙
import pandas as pd

# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\data
# header가 3번째 줄(인덱스 2)에 있을 때
data = pd.read_csv("data/data.csv", header=2)

print(data.head())
print(data.info())

print("height 결측치 :", data["height"].isnull().sum())
print("weight 결측치 :", data["weight"].isnull().sum())

#데이터프레임 전체의 결측치 확인
print(data.isnull().sum())

mean_height = data['height'].mean()
mean_weight = data['weight'].mean()

#fillna(대체값, inplace=True) inplace=True이면 원본데이터 교체
#그게 아니면 반드시 반환받아야 한다.
data['height'].fillna(mean_height, inplace=True)
data['weight'].fillna(mean_weight, inplace=True)

print("누락데이터 교체 후")
print(data.isnull()sum())

