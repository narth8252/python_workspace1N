import pandas as pd

#header가 3번째줄에 있음
data = pd.read_csv("./data/data.csv")

print(data.head)