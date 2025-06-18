#0513 9시pm.점프투파이썬 05-1-5.클래스의 상속
class Base: 
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        print("Base 생성자")

    def display(self):
        print(f"x={self.x} y={self.y}")

    #3.어거지이지만 이해위해 아래3가지 함수추가
    #파이썬은 오버로딩 불가(같은이름의 함수를2개생성X)
    #매개변수랑 함수를 가지고 구분하는 것이기때문에 파이썬은 오버로딩비허용
    #동일한이름의 함수가 한클래스에 존재할수없게 약속
    #부모자식간에는 지원하는 것을 오버라이딩
    #부모클래스에 만들면 자식이 호출?

#다형성: overloading - 동일클래스내에서 함수명같지만 형태다른 함수 만들수있는 성격
#                     def myadd(x,y)  def myadd(x,y,z)
#                     파이썬,자바스크립트는 오버로딩X,대신 매개변수기본값이라는것을 통해 유사한결과
#                     def myadd(x=1, y=2, z=3) myadd(), myadd(10), myadd(10,20,..)
#       overriding - 부모클래스와 자식클래스간에 벌어진다. 
#                   부모클래스에 있는 메서드를 수정할때,부모클래스의 함수이름과 자식클래스의 함수이름이 같으면
#                   부모클래스의 함수를 가린다.
#                   doubleX 특정변수에 종속되므로 따로 오버라이딩X
#                   파이썬은 오버라이딩만 지원
    def add(self):
        return self.x+self.y
        
    def doubleX(self):
        return self.x * 2
        
    def doubleY(self):
        return self.y * 2

class Child(Base): #Base부모클래스 상속받음
    def __init__(self, x=0, y=0, z=0):
        super().__init__(x, y)
        self.z = z
        print("Child 생성자")

    def display(self):
        #super()=부모클래스.호출한것이고 그 뒤에 내가 만든코드붙일수있음
        #super().함수명()
        super().display()
        print(f"x= {self.x} y= {self.y} z= {self.z}")
        # print(f"z={self.z}")
        return #안써도 리턴됨.

    #다른언어의 경우에는 부모생성자 먼저 호출하고 자식생성자를 호출 그런데 파이썬든
    #부모생성자를 호출하는 방식으로 설계하는것이 바람직

# Base 클래스 인스턴스 생성 및 메소드 호출
p = Base()
p.display()  #출력: Base생성자, x=0 y=0

p = Base(4, 5)
p.display()  #출력: Base생성자, x=4 y=5

# 오버라이딩(부모함수가리기) Child클래스 인스턴스생성 및 메소드호출
c1 = Child(1, 2, 3)
c1.display()  #출력: Base생성자, Child생성자, x=1 y=2, x=1 y=2 z=3
print( c1.doubleX()) #출력 2 (부모클래스 함수호출해서 쓰는것임)
print( c1.doubleY()) #출력 4

"""
-super()함수는 자식클래스에서 부모 클래스의 메소드(주로 생성자)를 호출할 때 사용합니다123.
-Child 클래스 __init__에서 부모 생성자를 호출할 때 반드시 부모가 기대하는 인자를 맞게 전달해야 합니다.
-display 메소드도 오버라이드 할 수 있으며, 부모 클래스의 display를 호출하려면 super().display()를 사용합니다.
-클래스명과 메소드명 오타를 반드시 수정해야 정상 동작합니다.
"""

class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"안녕하세요, 저는 {self.name}입니다.")

class Student(Person):
    def __init__(self, name, major):
        super().__init__(name)
        self.major = major

    def introduce(self):
        print(f"저는 {self.major}학과의 학생입니다.")

s = Student("홍길동", "컴퓨터공학")
s.greet()      # "안녕하세요, 저는 홍길동입니다." 출력
s.introduce() # "저는 컴퓨터공학학과의 학생입니다." 출력

"""상속과정의 그림
graph LR
    A[Person(부모 클래스)]
    B[Student(자식 클래스)]

    A-->|상속|B
    B-->|속성|name
    B-->|속성|major
    B-->|메소드|greet
    B-->|메소드|introduce
"""
"""
    상속의 이점
상속은 코드구조 단순화, 코드 재사용성을 높임. 
-코드의 재사용: 부모 클래스의 코드를 자식 클래스에서 그대로 사용할 수 있습니다3.
-계층적 구조: 클래스 간의 계층적 구조를 만들 수 있어, 프로그램의 구조를 더 명확하게 표현할 수 있습니다4.
    상속의 유형
-단일 상속: 한 클래스가 다른 클래스 하나만 상속받는 경우입니다.
-다중 상속: 한 클래스가 여러 다른 클래스를 상속받는 경우입니다14.
    MRO (Method Resolution Order)
다중 상속 시 메소드 호출 순서를 결정하는 방식입니다. 
클래스명.mro()로 확인할 수 있습니다.
"""
#파이썬에서 클래스 상속을 사용하는 대표적인 예시는 스타크래프트 유닛예제
#상속과 다중상속의 개념설명, 실생활에서 어떻게 적용할지 보여줍니다.
class Unit:
    def __init__(self, name, hp):
        self.name = name
        self.hp = hp

    def move(self):
        print(f"{self.name} 이동 중")

class AttackUnit(Unit):
    def __init__(self, name, hp, damage):
        super().__init__(name, hp)
        self.damage = damage

    def attack(self):
        print(f"{self.name} 공격력 {self.damage}")

class Flyable:
    def __init__(self):
        pass

    def fly(self):
        print(f"{self.name} 비행 중")

class FlyableAttackUnit(AttackUnit, Flyable):
    def __init__(self, name, hp, damage):
        super().__init__(name, hp, damage)

vulture = FlyableAttackUnit("벌처", 80, 5)
vulture.move()      # "벌처 이동 중" 출력
vulture.attack()    # "벌처 공격력 5" 출력
vulture.fly()       # "벌처 비행 중" 출력

# Unit 클래스는 기본 유닛의 속성과 메소드를 정의합니다.
# AttackUnit 클래스는 Unit을 상속받아 공격 능력을 추가합니다.
# Flyable 클래스는 비행 능력을 정의합니다.
# FlyableAttackUnit 클래스는 AttackUnit과 Flyable을 다중 상속받아 비행 공격 유닛을 구현합니다.