# 빈번한문법은 아니지만 프레임워크할때 쓰는 과정임.
# class MyClass: #클래스자체를 대상으로 동작하는 메서드(객체가 아닌 클래스에 영향)
#     #클래스변수의 영역이다
#     #객체만들던말던 한번만 만든다.
#     #생성자에서 이 부분을 건드리면 안된다.
#     count = 0  #객체가 만들어질때마다 몇개 만들어졌는지 확인
#     @staticmethod
#     def addCount():
#         # 클래스 변수에 접근하려면 클래스 이름으로 직접 접근해야 함
#         MyClass.count += 1
#     @classmethod
#     def increase(cls):  
#         cls.count += 1  # 클래스 메서드는 cls로 접근 가능

# 클래스 메서드 호출
# MyClass.increase()
# print(MyClass.count)  # 출력: 1
#     @classmethod 
#     def increase(cls): #cls는 클래스다
#         cls.count += 1 #당연히 에러발생
# MyClass.increase()
# print(MyClass.count )

# print( MyClass.addCount() ) #에러확인count += 1,
#앞에 self도 없고,count를 가리킬수없음. 변수공유없이 기능만쓰기위해 쓰는거라서
#반드시 함수내에서있는것만 써야돼. 함수밖에 있는거는 못써먹음

# @classmethod #←데코레이터
# def method_name(cls, ...):#cls← 클래스 자신을 의미 (self는 인스턴스)
#     ...

#생성자 __init__는 객체만들어질때마다 실행됨
#SelfCount.__count +=1 → 객체생성 때마다 1씩 증가.
class SelfCount:
    #변수앞에 __를 붙이면 외부에서 접근불가(private권한부여,은닉성)
    __count = 0 #private 클래스변수 선언
    #생성자에서 값증가시키면 됨
    def __init__(self):
        SelfCount.__count += 1
        print(SelfCount.__count) #클래스밖에서안되지,클래스내에선 출력가능

    @classmethod
    def count_output(cls):
        print(cls.__count)

s1 = SelfCount()         # 출력: 1
SelfCount.count_output() # 출력: 1
s2 = SelfCount()         # 출력: 2
SelfCount.count_output() # 출력: 2
s1 = SelfCount()         # 출력: 3
SelfCount.count_output() # 출력: 3
s1 = SelfCount()         # 출력: 4
SelfCount.count_output() # 출력: 4
#print( SelfCount.__count) #이 속성을 볼수없다는 에러발생


