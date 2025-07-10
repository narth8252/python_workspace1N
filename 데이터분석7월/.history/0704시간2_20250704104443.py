# C:\Users\Admin\Documents\딥러닝2507\250701파이썬데이터분석(배포X)\14차시_백현숙_250704
# 250704 am 10:30 쌤PPT.31p
import calendar

start_day, end_day = calendar.monthrange(2025, 7)
print(start_day) #1
print(end_day)   #31

for i in range(1, 13):
    start_day, end_day = calendar.monthrange(2025, i)
    print(f"{i}달의 마지막 날은 {end_day}일 입니다.")
"""
1달의 마지막 날은 31일 입니다.
2달의 마지막 날은 28일 입니다.
3달의 마지막 날은 31일 입니다.
4달의 마지막 날은 30일 입니다.
5달의 마지막 날은 31일 입니다.
6달의 마지막 날은 30일 입니다.
7달의 마지막 날은 31일 입니다.
8달의 마지막 날은 31일 입니다.
9달의 마지막 날은 30일 입니다.
10달의 마지막 날은 31일 입니다.
11달의 마지막 날은 30일 입니다.
12달의 마지막 날은 31일 입니다.
"""

#34p.시계열데이터 – 딥러닝때 사용함 
import pandas as pd 
import numpy as np
#날짜 문자열은 연도, 월, 일이 구분이 되면 된다. 
date_str = ["2019-10-01", "2019-10-02", "2019-10-03", "2019-10-04"]
idx = pd.to_datetime(date_str, infer_datetime_format=True)

print(type(idx)
print(idx)

