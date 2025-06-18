#0507 5시pm
#전에는 class 몰라도 지장없었는데, 3~4년전 Keras라는 사람이 모두 class사용(객체지향), 그래서 앞으로는 Keras 맞춰서 갈것임.

class Book:
    title="채식주의자"
    def __init__(self, title="쌍갑포차", paramprice=10000): 
        #b1, b2, b3...가 self에 들어와서 b1 b2 b3...의 타이틀이자 주소를 가지고 다님.
        #첫번째 인자에 객체주소를 가지고 다니자. self가 b1 b2 b3이다
        #첫번째 인자는 무조건 self.변수명 암묵적룰(self.쓰는 순간 객체구나.함.)
        #self.가 없으면 init라는 지역변수로 인식해서 에러는 안나지만 안돌아감
        print("self", self)  #객체주소(가상임.) #출력시 object at 주소임
        self.title = title       #self를 이용해서
        self.price = paramprice  #price는 지역변수로 함수가 끝나면 사라짐.
        self.count=10
        self.process()  #함수도 매개변수로 self를 받아가야한다.

    def process(self):
        self.total_price = self.price * self.count

    def output(self):
        print(self.title, self.price * self.count)
        
    pass

#title이라는 클래스 내부 변수에 접근하려면 접근연산자로 .(도트)
#파이썬은 title이라는 클래스에 b1 b2 b3 가 접근한다.
#자바처럼 각title클래스에 b를 1개씩 넣고 싶으면 ...
b = Book() #객체가 만들어진다.
print(b)
print(b.title)
print(b.price)  #b -> self로 전달

b2 = Book("아 지갑놓고 나왔네")  #웹툰추천
print(b2)
print(b2.title)

b3 = Book("뽀짜툰", 30000)
print(b3)
print(b3.title)

#함수는 일하는애라서 저장하는 애가 아님
#그래서 앞에 self.가 붙고 뒤에 함수명
#변수명 b1 b2 b3 ... 가 함수를 가져다 일함.
#출력시 object at 주소임: 메서드가 객체에 속하지X, 클래스메서드, static