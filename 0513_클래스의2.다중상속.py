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

#예시
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
파이썬에서 다중 상속을 사용하는 대표적인 예시는 스타크래프트 유닛을 구현하는 것입니다. 이 예제는 공격 유닛과 비행 유닛의 특성을 조합하여 공중 공격 유닛을 만드는 방식으로, 다중 상속의 개념을 잘 설명합니다.

예제: 스타크래프트 유닛
Unit 클래스: 기본 유닛의 속성과 메소드를 정의합니다.
AttackUnit 클래스: Unit을 상속받아 공격 능력을 추가합니다.
Flyable 클래스: 비행 능력을 정의합니다.
FlyableAttackUnit 클래스: AttackUnit과 Flyable을 다중 상속받아 비행 공격 유닛을 구현합니다.
"""

class Unit:
    def __init__(self, name):
        self.name = name

    def move(self):
        print(f"{self.name} 이동 중")

class AttackUnit(Unit):
    def __init__(self, name, damage):
        super().__init__(name)
        self.damage = damage

    def attack(self):
        print(f"{self.name} 공격력 {self.damage}")

class Flyable:
    def __init__(self, flying_speed):
        self.flying_speed = flying_speed

    def fly(self, name, location):
        print(f"{name} : {location} 방향으로 날아갑니다. [속도 {self.flying_speed}]")

class FlyableAttackUnit(AttackUnit, Flyable):
    def __init__(self, name, damage, flying_speed):
        AttackUnit.__init__(self, name, damage)
        Flyable.__init__(self, flying_speed)

vulture = FlyableAttackUnit("벌처", 5, 100)
vulture.move()      # "벌처 이동 중" 출력
vulture.attack()    # "벌처 공격력 5" 출력
vulture.fly("벌처", "북쪽")  # "벌처 : 북쪽 방향으로 날아갑니다. [속도 100]" 출력

# FlyableAttackUnit 클래스는 AttackUnit과 Flyable 클래스를 다중 상속받아, 비행 공격 유닛의 특성을 모두 구현할 수 있습니다.
# 이 예제는 다중 상속의 개념을 활용하여 코드의 중복을 줄이고, 특정 유닛의 여러 가지 기능을 효율적으로 구현하는 방법을 보여줍니다.
# 이처럼 다중 상속은 여러 클래스의 속성과 메소드를 하나의 클래스로 결합하여, 복잡한 시스템을 쉽게 관리할 수 있는 강력한 도구입니다.