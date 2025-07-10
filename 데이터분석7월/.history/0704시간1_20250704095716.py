# C:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)\14차시_백현숙_250704
# 250704 am 9시
import datetime

today = datetime.datetime.now() #현재날짜와 시간정보를 준다
print(today)

#파이썬의 dir함수가 있음, 기본적으로 내부구조를 보여준다
print(dir(datetime))

#날짜와 날짜의 연산 수행
d1 = datetime.date(2025,7,31)
print(d1)

from datetime import date
value = 1567345678 #타임스탬프
timestamp =  date.fromtimestamp(value)
print("date=", timestamp)