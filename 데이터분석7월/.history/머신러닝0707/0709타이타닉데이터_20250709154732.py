#250709 pm4시 train_and_test2.csv
#C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\머신러닝0707
#타이타닉데이터(결측치, 이상치처리, 스케일링, 서포트벡신)

import pandas as pd
import numpy as np

#데이터 불러오기
df = pd.read_csv('train_and_test2.csv')
#데이터 기본정보 확인
print("--- 데이터 정보 ---")
print(df.info())

#불필요한 컬럼제거(승객ID)
df = df.drop('PassengerId')




