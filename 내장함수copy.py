#0513 3시pm 점프투파이썬05-5내장함수
#all
print(abs(-4))  #4 #abs함수 ㅣ절대값ㅣ구하기
print(abs(4))   #4
print("---all 숫자형-----")
print( all( [1,2,3] ))  #True
print( all( [1,2,3,0] )) #0, False, "" :False 
                        # 요소전체가 True일때만 True    
print("---all 문자열-----")
print( all( ["a","b","c"] ))
print( all( ["a","b",""] ))

print("---any 숫자형-----")
print( any( [1,2,3] )) #any() 0이아닌것이 하나라도 있으면Fale
print( any( [1,2,3,0] ))

print("---any 문자열-----")
print( any( ["a","b","c"] ))
print( any( ["a","b",""] ))

print("---chr유니코드-----")
print(chr(97))     #a
print(chr(44032))  #가
# 전세계문자를 컴퓨터에서 설계하게 숫자로표현한 표준코드
# ord(c) : 문자를 유니코드 숫자로
# chr(n) : 유니코드 숫자를 다시 문자로
# 'a'는 97, 'A'는 65 → 소문자에서 32를 빼면 대문자
# if ord('a') <= ord(c) <= ord('z'): c가 소문자일 때만 변환

print("---dir(리스트)-----")
print( dir([1,2,3]))
print( dir(dict()))

print("---(튜플)-----")


print("---for i in enumerate(리스트)-----")
for i, name in enumerate(['body','foo','bar']):
    print(i, name)
    #출력: 0 body \n 1 foo \n 2 bar



print("---eval(expression)-----")
eval("1+2")         #3
eval("'hi' + 'a'")  #hia

result = eval('1+10+3') #14
print(result)

result = eval('(1+10+3)*3') #42
print(result)

print("---★filter(lambda x: 반복_가능한_데이터)-----")
#음수만 filter의 첫매개변수는 함수여야함
#두번째 매개변수로 전달된 요소하나를 매개변수로 하고 반환은 True/Flase
a=[3,4,5,-1,15,21,7,8,9]

def isPositive(x):
    if x>0: #양수면
        return True
    return False

poList = list(filter(isPositive, a))
print(poList) #[3, 4, 5, 15, 21, 7, 8, 9]

#위에처럼 하면 귀찮으니까
#일시함수 lambda를 만듬:한번쓰고 버림.한줄만 허용.
#플루터가 람다를 그렇게 많이 사용해서 만듬
poList = list(filter(lambda x: x>0, a))
print(poList)

#id 많이안씀
print("---★input([prompt])------")
#map,
print("---max(iterable)------")
#open, ord
print("---pow(수, 제곱할수)------")

#open, ord
print("---반올림round(수[,ndigits])------")

#점프투파이썬05-6표준라이브러리-Time 0513 3:40pm 
print("--점투파05-6표준라이브러리------")
print("--시간계산,★날짜는DB사용多------")
#컴퓨터는 시간
#문제:+d데이 계산
import datetime
day1 = datetime.date(2021,12,14)
day2 = datetime.date(2023,4,5)
print(day1) #2021-12-14
print(day2) #2023-04-05

#timedata객체로 바뀌고 날짜를 갖고있다ㅏ.ㅏ
day3 = day2 - day1
print(day3) #튜플로출력 477 days, 0:00:00
print(day3.days) # 477

#문제. 오늘은5/13 말일까지 며칠남았나?
import datetime
day4 = datetime.date(2025,5,13)
day5 = datetime.date(2025,5,30)
day6 = day5 - day4
print(f"5월말일까지 {day6.days}일 남았습니다.") #17

#연도,월 주면 말일계산해주는 calcalendar.라이브러리가 있다.
#마감까지 며칠 남았습니다. 할때 뿌려줌.
import calendar
from datetime import date

today = date.today()
year = today.year
month = today.month
#tuple반환 해당월의 첫날과 마지막날
last_day = calendar.monthrange(year, month)[1]
print(last_day) #31 출력

day1 =  datetime.date(year, month, last_day)
print( ( day1 - today).days) #타임델타에는 .days라는 ㅈ

print("---timedelta(.------")
from datetime import date

# 오늘 날짜
today = date.today()

# 특정 날짜
day1 = date(2025, 5, 30)

# 두 날짜 사이의 일수 계산
diff = day1 - today

# 일수 출력
print(diff.days)

"""
    timedelta 클래스
day1과 today 사이의 일수를 계산할 때 사용하는 timedelta 클래스의
.days 속성은 timedelta 객체에서 일수를 반환합니다.
timedelta 클래스는 datetime 모듈에서 기간을 나타내는 데 사용됩니다. 두 datetime 객체 간의 차이를 계산할 때 반환되는 객체가 바로 timedelta입니다.
이 객체는 일(days), 초(seconds), 마이크로초(microseconds)로 구성
"""

print("---요일(weekday)------")
#요일은 datetime.date객체의 weekday함수를 사용하면 구할수 있다.
#문제: 오늘이 무슨요일? 문자열로 받아라
print(today.weekday())
#숫자 0 1 2 3 4 5 6 (월 화 수 목 금 토 일)을 말함
#츨력"2025-04-11"

print("---요일(strptime 함수):쌤풀이------")
#   strptime 함수
#문자열을 datetime 객체로 변환. 문자열을 특정 형식으로 파싱하여 datetime 객체를 반환합니다. 예를 들어, 문자열 "2025-05-13"를 datetime 객체로 변환하려면 다음과 같이 사용할 수 있습니다.
day1 = datetime.datetime.strptime("2025-05-11", "%Y-%m-%d")
#datetime안에 datetime라이브러리 안에 strptime함수를 씀
print(day1.weekday())

def getWeekday(s):
    day1 = datetime.datetime.strptime(s, "%Y-%m-%d")
    weekday = day1.weekday()
    titles=["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return titles[weekday]
print(getWeekday("2025-04-11"))
#이런함수가 유틸클래스에 묶어놓고 static메서드로 들어가면 여러사람공유돼서 좋음.


print("---오늘요일 숫자로 가져옴------")
from datetime import date
print(today.weekday()) #1 (화요일)

print("---오늘요일 숫자로 가져온걸 문자열로 출력-----")
from datetime import date
week_day = ['월', '화', '수', '목', '금','토','일']
now = datetime.datetime.now()
day_num = now.weekday()

#요일숫자를 문자열로 변환
day_str = week_day[day_num]
print(f"오늘은 {day_str}요일입니다.")


#sample은 섞는거라 얘보다는 shuffle 주로씀.

print("--shutil 딥러닝에 사용多------")
#파일을 복사copy하거나 이동
import shutil
shutil.copy("./0513_내장함수.py", "./내장함수copy.py")

import glob
filelist = glob.glob("c/*") #*필수
print(filelist) #프롬프트창에서 dir로 찾는 관점.

filelist = glob.glob("./*.py") #*필수
print(filelist) 

#os많이씀 환경변수
#내PC우클릭>속성>시스템속성>고급시스템설정>환경변수>시스템변수 보여줌
import os
print( os.environ)
print( os.environ ['PATH'])

#현재 내경로
print(os.getcwd())
