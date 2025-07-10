# C:\Users\Admin\Documents\딥러닝2507\250701파이썬데이터분석(배포X)\14차시_백현숙_250704
# 250704 am 10:30 쌤PPT.38p~유용함 
#공부할때 가짜데이터 만들어야돼서 필요. 진짜데이터는 오류많고, 데이터가 많이 없음.
# 파일명 : exam14_6.py
#머신러닝,딥러닝할때 데이터생산능력
import pandas as pd 
import numpy as np

#2019년 9월 1일부터 2019년 9월 30일까지 생성
d = pd.date_range(start="2019-9-1", end="2019-9-30")
print(d)
d = pd.date_range(start="2019-1-1", periods=60)
print(d)
#2019년 1월중 일요일만 
d = pd.date_range(start="2019-10-1",  end="2019-10-30", freq='W')
print(d)

#2019년 각 달의 마지막날만 데이터 생성
np.random.seed(0)
ts = pd.Series(np.random.randn(12), 
    index=pd.date_range(start="2019-01-01", periods=12, freq="M"))
print(ts)

