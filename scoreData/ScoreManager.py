from ScoreData import ScoreData
#ScoreData.py파일에서 ScoreData클래스를 가져와라.
import pickle

class ScoreManager:
    def __init__(self): #생성자
        self.scoreList = [
            # ScoreData(),
            # ScoreData("조승연", 90, 80, 90),
            # ScoreData("강승환", 70, 60, 50),
            # ScoreData("함승원", 60, 50, 40),
            # ScoreData("함승원", 87, 75, 44),
            # ScoreData("이망원", 55, 55, 45)
        ]
    def printAll(self):
        for s in self.scoreList:
            s.print()   #ScoreData클래스의 def print(self):가 나오는것

    def menuDisplay(self):
        print("------------")
        print("    메뉴    ")
        print("------------")
        print("   1.추가   ")
        print("   2.출력   ")
        print("   3.검색   ")
        print("   4.수정   ") #0508_2시pm 수정,삭제,정렬 추가 #이름
        print("   5.삭제   ") #remove() 이름
        print("   6.정렬   ") #sort() #총점내림차순
        print("   7.저장   ")
        print(" 8.불러오기 ")
        print("   0.종료   ")
        print("------------")
    
    #0509 pm5시 모듈-pickle 배우고 추가해봄
    #맨위에 import pickle
    def save(self):  #파일이름.bin, 읽을지쓸지(write bin, read bin, txt파일읽어올꺼면 w/r)
        with open("score.bin", "wb") as f: #with open은 자동 f.close()
        #with "open()"기능을  f:안에 넣겠다.  = 과 같은의미:오픈만해라.
            pickle.dump(self.scoreList, f)
            #피클은 내장모듈중 하나 
            #피클안에 있는 덤프라는 함수를 불러옴
            #f안에 self.scoreList를 저장한다.
            #dump도 피클안에 내장되있는 함수
            #f가 score.bin파일을 열어놓고 있는 상태인데, 피클.덤프가 저장하는 기능을 함.
            #self.scorList안에 암거나 저장해도 됨.

    def load(self):
        with open("score.bin", "rb") as f:
            self.scoreList = pickle.load(f)
            #pickle안에 있는 load내장함수 를 쓴것인데
            #위에 저장한 f(오픈어쩌고)를 피클에 있는 load함수를써서 로딩해서
            #self.scoreList안에 저장하겠다.
        self.printAll()
    #피클모듈
    

    def append(self):
        sc = ScoreData() #객체생성
        sc.name = input("이름: ")
        sc.kor = int(input("국어 : "))
        sc.eng = int(input("영어 : "))
        sc.mat = int(input("수학 : "))
        sc.process()
        self.scoreList.append(sc)
        #scoreManager클래스 안에 scoreList안에만 추가했다.
        #score.bin에는 저장되지않았다.

    def search(self):
        name = input("찾을이름 : ")
        #filter는 두번째 매개변수로 전달된 list를 받아서
        #for문 돌려서 첫번째 매개변수로 전달되 함수를 호출
        #람다:매개변수하나(scoreList에 저장된 객체 하나)
        #     반환은 True/Flase

        #코드순서:5     4     1      3                   2
        # resultList = list(filter( lambda s in s.name, self.scoreList))
        #전체실행이 아니라 실행준비상태임.
        #for문을 사용하거나 list로 둘러싸면 list생성자가 호출되면서 filter가 모든작업 완료
        resultList = list(filter(lambda item: name in item.name,
                            self.scoreList))
        #데이터가 없을경우 처리 len(resultList) 데이터개수 반환
        if len(resultList) == 0:
            print("데이터가 없습니다.")
            return  #else쓰지말고 리턴으로 함수종료해.        
        #enumerate함수가 list를 전달하면 index와 객체tuple을 반환
        for i, s in enumerate(resultList):
            print(f"[{i}]", end="")
            s.print()
            #▽WeekpayManager에서 사용함
            # print(i, end ="\t") 
            # s.output()

    def modify(self):  #0508_3사pm 수정
        name = input("수정할 이름 : ")
        resultList = list(filter( lambda item: name in item.name, self.scoreList))
        if len(resultList) == 0:
            print("데이터가 없습니다.")
            return
        
        #   위치,요소  enu...가 ()안에 있는 위치와 요소를 보여준다.
        for i, s in enumerate(resultList):
            print(i, end ="\t")
            s.print()
        sel = int(input("수정할 대상은(번호) "))
        #수정대상의 참조 가져오기
        item = resultList[sel]  #rL에 있는 sel위치에 있는 요소를 item에 넣겠다.
        item.name = input("이름 : ")
        item.kor = int(input("국어 : "))
        item.eng = int(input("영어 : "))
        item.mat = int(input("수학 : "))
        item.process = () #다시계산 스코어데이타.py클래스안의 def process(self):사용해서 계산하겠다.

    def delete(self):  #0508_3:20pm 삭제
        name = input("삭제할 이름 : ")
        resultList = list(filter( lambda item: name in item.name, self.scoreList))
        if len(resultList) == 0:
            print("데이터가 없습니다.")
            return
        for i, s in enumerate(resultList):
            print(i, end ="\t")
            s.print()
        sel = int(input("삭제할 대상은(번호) "))
        self.scoreList.remove(resultList[sel])
        #resultList에서 위에self.scoreList.remove로 바꾼이유:
        # bin파일에서는 삭제하지않고, ScoreManager클래스 안에있는 리스트에서만 삭제해라.
        #remove는 객체참조를 직접부여. 그 객체참조를 찾아서 삭제
        #remove는 받아서 원주소를 삭제 - 실제 대상 삭제
        #del은 받아온주소만 삭제 - 명단만 삭제
        #

    def sort(self): #원본두고 정렬한 결과만 출력 sorted()
        #sort와 sorted의 차이:
        # sorted(self.scoreList, key=) 
        # self.scoreList는 가만히 두고, 정렬된(sorted) 리스트를 resultList에 저장하겠다 
        # #key=에 전달할해야할 람다는 매개변수하나, 반환값정렬할수있는 데이터타입
        # #>  < 연산자 가능
        # s1 = ScoreData()
        # s2 = ScoreData()
        # s1 > s2
        #파이썬제공하는 기본타입 int,float,str..
        #크다적다 말할 수 있으면 쓸수有
        # 6            1      2              3    4      5            
        resultList = sorted(self.scoreList, key=lambda item : item.tot,
                            reverse=True ) #기본이 오름차순이라 내림차순하려고.
        #어떤식으로 정렬하겠다라는게 key다.ㄴ
        #scoreList의 요소가 item인데 scoreData.py안에 class ScoreData를 의미.
        #lambda x: x == 1 은 아래2줄과 같은의미이다. 람다는 한줄짜리 함수.
        #def lambda (x):
        #   return x == 1
        """
        # a = [3,5,2,14,5,2]
        # #a.sort()    # a 리스트 자체가 변경돼요
        # print(a)

        # b = sorted(a)
        # print(a)
        # print(b)

        # lambda x: x == 1

        # print(sorted(a, key=lambda x: x))
        #a리스트 안에있는 x라는 요소 하나하나를 x라는 변수에 넣고,
        #  3,5,2,14,...를 키를 기준으로 해서 정렬하겠다.
        #for x in a: (람다의 x: 과 for문의 x의미와 같다고 이해하라.)
        # # def lambda(item):
        # #     return item
        """
        #7
        for i in resultList:
            i.print()

    def start(self):
        #함수주소를 배열에 저장하고 호출.
        funcList =[None, self.append, self.printAll, 
                   self.search, self.modify, self.delete, self.sort,
                   self.save, self.load]
#위에함수추가後 마지막에 검색    수정      삭제    정렬  피클로저장  불러오기
        while True:
            self.menuDisplay()
            choice = int(input("선택 : "))
            if choice>0 and choice<len(funcList):
                funcList[choice]()
            elif choice==0:
                return
            else:
                print("잘못된 메뉴입니다.")


if __name__ =="__main__":
    sm = ScoreManager()
    # sm.printAll()
    sm.start()
    # sm.modify()
