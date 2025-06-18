""" 0513 10시am
-다중상속
파이썬은 다중상속허용
클래스를 여러개 상속받는 경우
A >B > C 중첩상속 : 모든언어가 이 구조 허용

        A  B
        C     부모클래스가 A,B인경우 2개이상의 클래스를 상속받는경우
                다이아몬드상속
    A > B --->D
    A > C --->D
             java는 단일상속만가능
"""
# 부모 클래스 1
class Flyable:
    # def __init__(self, name):
    #     self.name = name

    def fly(self):
        print(f"날수있다") 
    
    def walk(self):
        print(F"두다리로 걷는다") 

# 부모 클래스 2
class Swimmable:
    # def __init__(self, company):
    #     self.company = company

    def swim(self):
        print(f"수영할수있다.") 

    def walk(self):
        print(F"***두다리로 걷는다***")

# 다중 상속을 받는 자식 클래스
class Duck( Swimmable, Flyable): #
    # def __init__(self, name, company, position):
    #     Human.__init__(self, name)
    #     Worker.__init__(self, company)
    #     self.position = position

    def quak(self):
        print(f"꽥꽥.")

# 인스턴스 생성 및 메소드 호출
d1 = Duck()
d1.fly()          # fly클래스 메소드 호출 #날수있다
d1.swim()           # swim클래스 메소드 호출  수영할수있다.
d1.quak()  # quak클래스 메소드 호출 #꽥꽥.
d1.walk() #메서드명동일한경우 앞에거 먼저 호출 #***두다리로 걷는다***
 #class Duck( Swimmable, Flyable): 

print( Duck.__mro__ )
#(<class '__main__.Duck'>, <class '__main__.Swimmable'>, <class '__main__.Flyable'>, <class 'object'>)
#(.__mro__)함수:같은함수가 있으면 덕이 최우선, 마지막이 

"""
# 부모 클래스 1
class Human:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"안녕하세요, 저는 {self.name}입니다.")

# 부모 클래스 2
class Worker:
    def __init__(self, company):
        self.company = company

    def work(self):
        print(f"{self.company}에서 열심히 일합니다.")

# 다중 상속을 받는 자식 클래스
class Employee(Human, Worker):
    def __init__(self, name, company, position):
        Human.__init__(self, name)
        Worker.__init__(self, company)
        self.position = position

    def show_position(self):
        print(f"직급은 {self.position}입니다.")

# 인스턴스 생성 및 메소드 호출
emp = Employee("홍길동", "삼성전자", "과장")
emp.greet()          # Human 클래스 메소드 호출
emp.work()           # Worker 클래스 메소드 호출
emp.show_position()  # Employee 클래스 메소드 호출
"""