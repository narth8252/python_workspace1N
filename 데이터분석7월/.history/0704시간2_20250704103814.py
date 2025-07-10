# C:\Users\Admin\Documents\딥러닝2507\250701파이썬데이터분석(배포X)\14차시_백현숙_250704
# 250704 am 10:30
import calendar

start_day, end_day = calendar.monthrange(2025, 7)
print(start_day) #1
print(end_day)   #31

for i in range(1, 13):
    start_day, end_day = calendar.monthrange(2025, i)
    print(f"{i}달의 마지막 날은 {end_day}일 입니다.")




