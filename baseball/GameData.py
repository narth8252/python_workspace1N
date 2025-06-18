#클래스(객체지향)으로 야구게임만들기_GameData.py 0508 4시pm
import random

class Baseball:
    def __init__(self):
        self.computer = [-1,-1,-1,-1] #아무값도 할당없다. 로 시작할때 맨앞-1 사용(안써도되는데)
        self.person = [-1,-1,-1,-1] #아무값도 할당없다. 로 시작.
        self.count=0 #몇번했는지 저장
        self.personList =[]

    def init_computer(self):
        cnt=1 #count 랜덤값3개를 추출해야하는데 중복되면 안됨
        while cnt<=3:
            v = random.randint(0,9)
            if v not in self.computer: #없을때 중복안될때
                self.computer[cnt]=v
                cnt +=1  #있을때 중복될때는 계속 돌면됨, 4일때 나가면됨.
        #for문으로 쓰면 if쓰고 어쩌고 해야되는데 while문으로는 여러번안해도 됨.

    def init_person(self): #숫자3개 입력 편하게 받으려면? 문자열1 2 3보다 스페이스바가 편하겠지?
        s = input("숫자3개를 입력하세요(0~9사이의숫자,예시:0 1 2)")
        numberList = s.strip().split(" ") #.strip()함수는 입력자가 스페이스바눌러서 나오는 공백 없애줌
        #  2             1            ""사이에 공백안하면 에러남
        self.person[1]=int( numberList[0]) #3줄정도면 for문말고 노가다해도 됨.
        self.person[2]=int( numberList[1])
        self.person[3]=int( numberList[2])

    def getResult(self): #튜플이라 통으로줘도됨
        #스트라이크,볼,아웃 개수
        strike=0
        ball=0
        out=0
                 
        for i in range(1, 4):
            if self.person[i] in self.computer:
                if self.computer[i] == self.person[i]:
                    strike+=1  #그러면 스트라이크
                else:
                    ball+=1  #그러면 볼
            else:
                out+=1
        return strike, ball, out
    
    def start(self):
        #3스트라이크이거나 5번기회를 다 쓰면 종료한다. for문, while,
        flag = False #아직3strike아님을 나타내기위한 변수 flag
        self.init_computer()
        print(self.computer) #개발자가 잘하고있는지 알려주기위한 컨닝페이퍼
        while flag ==False and self.count<=5:
            self.init_person()
            strike, ball, out = self.getResult()
            print(f"strike:{strike} ball:{ball} out:{out}")
            #5시pm dict으로 추가
            self.personList.append({"person":[x for x in self.person],
                                   "strike":strike,"ball":ball,"out":out})
            if strike ==3:
                flag=True #while끝나고 빠져나가면서 빵빠레를 함수안에서 종료해줘야 이쁨
            self.count+=1
        #while문이랑 flag짝꿍. for문이랑break써도 됨.쌤이 flag에 익숙.

#일단 getResult만들기 전에 여기까지해서 테스트하자
if __name__=="__main__":
    b = Baseball()
    # b.init(b.computer) 
    # b.init(b.person)
    # print(b.computer)
    # print(b.person)
    # print(b.getResult())  # getResult만들고 테스트하자
    #만들면서 확인해보고 주석으로 막고 최종프린트
    b.start()

