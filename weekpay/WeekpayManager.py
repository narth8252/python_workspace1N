#0507 3시pm 폴더weekpay 만들어서 쪼개기 
#객체지향 class
#모듈 Weekpay.py 파일에 있는 클래스 Weekpay를 가져오겠다.(.py 생략)  

#####파일명         클래스명
from weekpay import Weekpay #외부파일(모듈을) 이 파일로 불러오기

class WeekpayManager:

    def __init__(self):
        self.wList = [
            Weekpay("홍길동", 20, 20000),
            Weekpay("고길동", 10, 50000),
            Weekpay("김길동", 40, 40000),
            Weekpay("장길동", 30, 10000)]

    def output(self):
        for w in self.wList:
         w.output()

    def search(self):
        name = input("찾을이름 : ")
        resultList = list(filter(lambda w : w.name in w.name, self.wList))  #0507 4시pm #w.name == name 1명씩찾아
        if len(resultList) == 0:       #0507 4:20pm
            print("데이터가 없습니다.")
            return   #else보다 return이 코드가 이쁨
        
        #resultList[0].output() #1명일때 쓸수있음. #Weekpay의 output
        for w in resultList:
            w.output()     #공무원일할때는, 아이디 중복확인시 아이디가 없습니다. 라고 원하는 틀이 정해져있음. 4:25pm

    def modify(sellf):
        name = input("찾을이름 : ")
        resultList = list(filter(lambda w : w.name in w.name, self.wList))  #0507 4:45pm #w.name == name 1명씩찾아
        if len(resultList) == 0:       
            print("데이터가 없습니다.")
            return   #else보다 return이 코드가 이쁨
        
        #enumerate 함수는 인덱스랑 데이터를 한꺼번에 반환
        #resultList[0].output() #1명일때 쓸수있음. #Weekpay의 output
        for i, w in enumerate:
            print(i, end ="\t")
            w.output()   
            
        sel = int(input("수정할 대상을 입력하세요(숫자로)"))
        temp = resultList[sel]
        temp.name = input("이름 : ")
        temp.work_time = input("근무시간")
        temp.per_pay = input("시급 ")
        temp.process()

    def start(self):
      print("start")

if __name__ =="__main__":
    mgr = WeekpayManager()
    # mgr.output()
    # mgr.search()
    mgr.modify()
    mgr.output()
