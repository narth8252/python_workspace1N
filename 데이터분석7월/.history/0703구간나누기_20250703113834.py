#쌤PPT-2p.
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

#꼭써야하는건 아니고 직접줘도 됨.5개를..
data.dropna(subset=['power'], axis=0, inplace=True)
count, bin_dividers = np.histogram(data['power'], bins=4)
print("각 구간별 데이터 개수 : ", count)
print("구간정보 : ", bin_dividers)

#bin_dividers=np.arry([40,120,140,200,300]) 직접부여가능
bin_nsmr = ["D", "C", "B", "A"]
data["grade"] = pd.cut(x=data['power'],
                       bins= bin_dividers,
                       labels=bin_names,
                       include_lowest=True)
print(data[:20])
print(data[])