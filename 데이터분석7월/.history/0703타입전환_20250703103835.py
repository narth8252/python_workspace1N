#쌤PPT-21p.
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)\11차시_백현숙\[평생]원고_v1.0_11회차_데이터셋_백현숙_0915_1차.pptx
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\data
# auto-mpg.csv
#파일명 : exam11_5.py
#데이터표준화
import pandas as pd 
import numpy as np 
data = pd.read_csv('./data/auto-mpg.csv')
print(data.info())
print(data.head())
 #타입이 맞지 않을 경우 전환을 해서 사용해야 한다 ⭐
#현재 사용하는 파이썬 버전은 문자열 데이터라도 수치 형태면 자동으로 수치자료로 처리한다 
#파이썬 버전에 따라 다르게 동작할 수 도 있다 
data.columns=['mpg', 'cyl', 'disp', 'power', 'weight', 'acce', 'model']
print(data.dtypes)
print(data.head())
 print( data['disp'].unique())
