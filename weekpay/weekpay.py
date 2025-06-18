class Weekpay:      

  def __init__(self, name="", work_time=20, per_pay=10000):
    self.name = name  #인스턴스 변수 name에 매개변수 name 값을 저장
    self.work_time = work_time
    self.per_pay = per_pay
    self.process()

  def process(self):
    self.pay = self.work_time * self.per_pay

  def output(self):
    print(f"{self.name} {self.work_time} {self.per_pay} {self.pay}")

#파이썬의 모듈들은 내장변수가 있다. __name__
print( __name__ )  #이 모듈로 직접 실행하면 __main__이 들어온다. python Weekpay.py
#import 되서 실행하면 파일명이 전달 python WeekpayManager.py

if __name__=="__main__":
    w1 = Weekpay("A")
    w1.output()
