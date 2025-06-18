#0513.10:20am

class A:
    def __init__(self):
        print("A생성자호출")

class B(A):
    def __init__(self):
        print("B생성자호출")
        super().__init__() #부모클래스 메서드호출
# super통해서 해야하는데 아래처럼 하면 A2번호출되는 에러가능성(현재파이썬은 해결됐으나)
        # A.__init__(self) #여튼 호출할땐 self필수
        
class C(A):
    def __init__(self):
        print("C생성자호출")
        super().__init__()

c = C()  #출력: C생성자호출 /n A생성자호출


print("------------------------")

class D(A):
    def __init__(self):
        print("D생성자호출")
        super().__init__()

d = D() #객체생성 __MRO규칙따름

#isinstance(객체, 클래스) 함수:중요
#이 객체가 클래스의 인스턴스인지 확인해줌 True/Flase
print(isinstance(d, A))
print(isinstance(d, B))
print(isinstance(d, C))
print(isinstance(d, str)) #str은 A하고 상관없으니까
print(isinstance(d, object))

#object : 모든클래스의 베이스라 파이썬내장, 무조건상속받음.
#개발자편하라고. 모든클래스는 object를 상속받게 해놈=자식임
#a.치면 엄청많은 함수뜸.

a = object()
print(a.__class__)
print(a.)


        