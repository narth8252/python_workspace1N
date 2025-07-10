# C:\Users\Admin\Documents\딥러닝2507\250701파이썬데이터분석(배포X)\14차시_백현숙_250704
# 250704 am 10:30 
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


