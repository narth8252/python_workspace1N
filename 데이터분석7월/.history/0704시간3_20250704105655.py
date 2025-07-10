# C:\Users\Admin\Documents\딥러닝2507\250701파이썬데이터분석(배포X)\14차시_백현숙_250704
# 250704 am 10:30 쌤PPT.38p~유용함 
#공부할때 가짜데이터 만들어야돼서 필요. 진짜데이터는 오류많고, 데이터가 많이 없음.
# 파일명 : exam14_6.py
#머신러닝,딥러닝할때 가짜 데이터생산능력
import pandas as pd 
import numpy as np

#2019년 9월 1일부터 2019년 9월 30일까지 생성
d = pd.date_range(start="2025-07-04", end="2025-09-23")
print(d)
d = pd.date_range(start="2019-1-1", periods=60)
print(d)
#2019년 1월중 일요일만 
d = pd.date_range(start="2019-10-1", end="2019-10-30", freq='W')
print(d)

d = pd.date_range(start="2025-07-04", periods=30, freq="T")
print(d)

#2019년 각 달의 마지막날만 데이터 생성
np.random.seed(0)
ts = pd.Series(np.random.randn(12), 
    index=pd.date_range(start="2025-07-04", periods=30, freq="T"))
print(ts)

#period필드와freq프리퀀시 둘이 결합해서 일,시간
d = pd.data_range(start="2025-07-04", periods=30, freq=)
#가짜데이터 생성방법
#np.random.randm 함수는 가우스부노를 따르는 값을 랜덤하게 생성한다.
#가우스분포를 따르는 실수값이 중요한잉
#머신러닝을 만든 통계학자들이 자연계에서 얻어지는 모든값을 분석함
#대부분의 경우 양극단으로 갈수록 작아지고, 중간값으로 갈수록 커지는 종모양 차트가 만들어지더라
#->이게 바로 정규분포. 여기서 출발

