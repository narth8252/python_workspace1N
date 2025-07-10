import pandas as pd

# header가 3번째 줄(인덱스 2)에 있을 때
data = pd.read_csv("./데이터분석250701딥러닝/data/data.csv", header=2)

print(data.head())
print(data.info())
