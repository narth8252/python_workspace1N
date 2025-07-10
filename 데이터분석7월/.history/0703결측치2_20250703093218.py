import pandas as pd

#header가 3번째줄에 있음
data = pd.read_csv("./데이터분석250701딥러닝/data/data.csv")

print(data.head())
print(data.info())
