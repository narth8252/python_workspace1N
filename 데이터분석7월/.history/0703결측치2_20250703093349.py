import pandas as pd

# header가 3번째 줄(인덱스 2)에 있을 때
data = pd.read_csv("data.csv", header=2)

print(data.head())
print(data.info())
