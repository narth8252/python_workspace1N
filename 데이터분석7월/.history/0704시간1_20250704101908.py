# C:\Users\Admin\Documents\딥러닝2507\250701파이썬데이터분석(배포X)\14차시_백현숙_250704
# # 250704 am 9시
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
timestamp =  date.fromtimestamp(value) #타임스탬프로부터 날짜를 얻어낸다.
print("date=", timestamp)

from datetime import datetime, timedelta

# 오늘로부터 3일 후
current = datetime.today()  # 현재 시간과 날짜
after = current + timedelta(days=3)
print("현재시간 : ", current)
print("3일후 : ", after)

# 오늘로부터 3일 하고 4시간 뒤
current = datetime.today()  # 현재 시간과 날짜
after = current + timedelta(days=3, hours=4)
print("현재시간 : ", current)
print("3일후 : ", after)

# 오늘로부터 3일전 시간은 4시간후
current = datetime.today()  # 현재 시간과 날짜
after = current + timedelta(days=-3, hours=4)
print("현재시간 : ", current)
print("3일전 : ", after)

# 오늘로부터 2주전
current = datetime.today()  # 현재 시간과 날짜
after = current + timedelta(weeks=-2)
print("현재시간 : ", current)
print("2주전 : ", after)

from datetime import datetime, timedelta
import pytz

# 한국 시간대 설정
KST = pytz.timezone('Asia/Seoul')

# 날짜 출력 포맷 함수
def format_datetime(dt, only_date=False):
    if only_date:
        return dt.strftime("%Y-%m-%d (%A)")  # 요일 포함
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S (%A)")

# 시간 계산 및 출력 함수
def show_time_diff(label, days=0, weeks=0, hours=0, only_date=False):
    current = datetime.now(KST)
    after = current + timedelta(days=days, weeks=weeks, hours=hours)
    
    print(f"▶ {label}")
    print("현재 시간 : ", format_datetime(current, only_date))
    print(f"{label} : ", format_datetime(after, only_date))
    print("-" * 40)

# ──────────────────────────────

# 예시 실행
show_time_diff("오늘로부터 3일 후", days=3)
show_time_diff("오늘로부터 3일 4시간 후", days=3, hours=4)
show_time_diff("오늘로부터 3일 전 + 4시간", days=-3, hours=4)
show_time_diff("오늘로부터 2주 전", weeks=-2)

# 날짜만 보고 싶을 경우 (시간 생략)
show_time_diff("3일 후 (날짜만)", days=3, only_date=True)
"""
▶ 오늘로부터 3일 후
현재 시간 : 2025-07-04 10:23:45 (Friday)
오늘로부터 3일 후 : 2025-07-07 10:23:45 (Monday)
----------------------------------------

▶ 오늘로부터 3일 4시간 후
현재 시간 : 2025-07-04 10:23:45 (Friday)
오늘로부터 3일 4시간 후 : 2025-07-07 14:23:45 (Monday)
----------------------------------------

▶ 오늘로부터 3일 전 + 4시간
현재 시간 : 2025-07-04 10:23:45 (Friday)
오늘로부터 3일 전 + 4시간 : 2025-07-01 14:23:45 (Tuesday)
----------------------------------------

▶ 오늘로부터 2주 전
현재 시간 : 2025-07-04 10:23:45 (Friday)
오늘로부터 2주 전 : 2025-06-20 10:23:45 (Friday)
----------------------------------------

▶ 3일 후 (날짜만)
현재 시간 : 2025-07-04 (Friday)
3일 후 (날짜만) : 2025-07-07 (Monday)
----------------------------------------

"""