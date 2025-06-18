
##0508 1시pm ScoreData.py 성적처리
##클래스로 야구게임, 성적처리 끌어오기 __intit__, __main__에서 끌어오기 해야해서 VS코드로 폴더 각각 만들어서 함.
#한사람정보 - 데이터베이스 레코드 하나
#파이썬의 경우는 파일명과 클래스명은 아무 관계없다.
class ScoreData:
  def __init__(self, name="홍길동", kor=100, eng=100, mat=100):
    self.name = name
    self.kor = kor
    self.eng = eng
    self.mat = mat
    self.process()

  def process(self):
    self.tot = self.kor + self.eng + self.mat
    self.avg = self.tot / 3

    if self.avg >= 90:
      self.grade = "수"
    elif self.grade>=80:
      self.grade = "우"
    elif self.grade>=70:
      self.grade = "미"
    elif self.grade>=60:
      self.grade = "양"
    else:
      self.grade = "가"

class ScoreData:
  def print(self):
    print(f"{self.name}", end="\t")
    print(f"{self.kor}", end="\t")
    print(f"{self.eng}", end="\t")
    print(f"{self.mat}", end="\t")
    print(f"{self.tot}", end="\t")
    print(f"{self.avg:.2f}", end="\t")
    print(f"{self.grade}", end="\n")

if __name__ == "__main__":
  s = ScoreData()
  s.print()


  # def output(self):
  #   print(f"{self.name} {self.kor} {self.eng} {self.mat} {self.tot} {self.avg} {self.grade}")

class Weekpay:
  def __init__(self, name="", hour=20, per_pay=10000):
    self.name = name
    self.hour = hour
    self.per_pay = per_pay
    self.process()

  def process(self):
    if self.hour <= 40:
      self.pay = self.hour * self.per_pay
    else:
      self.pay = 40 * self.per_pay + (self.hour - 40) * self.per_pay

  def output(self):
    print(f"{self.name} {self.hour} {self.per_pay}")


class WeekpayManager:
  def __init__(self):
    self.pay_list = [
        Weekpay("홍길동", 20,20000),
        Weekpay("홍길동", 20,20000),
        Weekpay("홍길동", 20,20000),
        Weekpay("홍길동", 20,20000)]

  def output(self):
    for w in self.wList:
      w.output()

manager = WeekpayManager()
manager.output()