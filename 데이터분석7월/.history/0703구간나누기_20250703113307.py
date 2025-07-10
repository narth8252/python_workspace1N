#쌤PPT-26p.
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)\11차시_백현숙\[평생]원고_v1.0_11회차_데이터셋_백현숙_0915_1차.pptx
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\data
# auto-mpg.csv
#파일명 : exam11_4.py
#구간분할
#구간나누기 코딩
#bins = 나눠야할 구간의 개수
#구간을 나눠서 각 구간별 데이터개수와 구간에 대한 정보를 바노한
import pandas as pd
import numpy as np

# 데이터 불러오기 (auto-mpg.csv 파일이 data 폴더에 있어야 함)
data = pd.read_csv('./data/auto-mpg.csv')
print(data.info())
print(data.head())

#타입이 맞지 않을 경우 전환을 해서 사용해야 한다 
#현재 사용하는 파이썬 버전은 문자열 데이터라도 수치 형태면 자동으로 수치자료로 처리한다 
#파이썬 버전에 따라 다르게 동작할 수 도 있다 
data.columns=['mpg', 'cyl', 'disp', 'power', 'weight', 'acce', 'model']
print(data.dtypes)

#잘못된 데이터를 NaN으로 먼저 바꾼다 
data['disp'].replace('?', np.nan, inplace=True)
data.dropna(subset=['disp'], axis=0, inplace=True)
data['disp'] = data['disp'].astype('float')


count, bin_dividers = np.histogram(data['power'], bins=4)
print("각 구간별 데이터 개수 : ", count)
print("구간정보 : ", bin_dividers)