import pandas as pd

import pandas as pd

# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\data
# header가 3번째 줄(인덱스 2)에 있을 때
data = pd.read_csv("data/data.csv", header=2)

print(data.head())
print(data.info())
