#0507 5시pm
class Book:
    title-"채식주의자"
    def __init__(self, title="쌍갑포차", paramprice=10000): 
        #b1, b2, b3...가 self에 들어와서 b1 b2 b3...의 타이틀이자 주소를 가지고 다님.
        #첫번째 인자에 객체주소를 가지고 다니자. self가 b1 b2 b3이다
        #첫번째 인자는 무조건 self.변수명 암묵적룰(self.쓰는 순간 객체구나.함.)
        #self.가 없으면 init라는 지역변수로 인식해서 에러는 안나지만 안돌아감
        self.title = title       #self를 이용해서
        self.price = paramprice  #price는 지역변수로 함수가 끝나면 사라짐.
        self.count=10
    def process(self):
        self.total_price = self.price * self.count
        
    pass

#title이라는 클래스 내부 변수에 접근하려면 접근연산자로 .(도트)
#파이썬은 title이라는 클래스에 b1 b2 b3 가 접근한다.
#자바처럼 각title클래스에 b를 1개씩 넣고 싶으면 ...
b = Book() #객체가 만들어진다.
print(b.title)
print(b.price)  #b -> self로 전달

b2 = Book()
print(b2.title)

b3 = Book()
print(b3.title)


