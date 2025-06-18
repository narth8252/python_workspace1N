#0512_4:20pm
#데코레이터:
    # 만들일은 드물지만 알고넘어가면 다음에 공부할때나 쓸일있을때 찾아보라고.
    #못만들어도 남들이 만들어논거 갖다 쓰면돼.
#개발자의 좋은 마음가짐
    #항상 해야되는 코드는 아니다
    #다른이들도 많이 안한다는 것임.
    #직접 만들지 않아도 기존에 있는 코드 가져와서 쓰면됨.
 
#@staticmethod처럼 @로 시작한다.
#함수의 전후동작을 감싸서 함수가로채기를 해서 작업호출하고 원래함수호출

#데코레이터는 함수안데 또 함수(중첩함수)
#매개변수로 받아서 중첩함수안에서 받아간 함수를 호출
#파이썬은 웹에서 주로사용
#반드시 로그온해야 접근되는 함수만들고 싶을때
#함수를 중간에 가로채서 실행시간 체크가능(개발자 누군가 느릴때..)
                        #매개변수는 중첩함수안에서 호출될 함수
def decorator1( func ):  # 데코레이터 함수: 원래 함수 func을 인자로 받음
    def wrapper():      # 중첩 함수: 원래 함수 대신 실행될 함수
        print("함수호출전")  # 기본문법.외워
        func()          
        print("함수호출후")
    return wrapper      # wrapper 함수(중첩함수)의 "참조"를 반환

@decorator1
def Hello(  ):  # 데코레이터 함수: 원래 함수 func을 인자로 받음
    print("Hello")  #호출하는순간 @decorator에게 납치돼서 호출됨

Hello()
#@decorator1에게 납치당함 → wrapper → func() 을 통해 함수호출
"""출력
함수호출전
Hello
함수호출후
"""

print("ˇ"*40)
#2.타임데코레이터@
#첨만들땐 이상해도 구조가문법이 정해져있어서 못바꾸니
#앞에 __있는건 함수가 자기이름 데리고다니는것임(리플렉션)
import time

def time_decorator( callback ):  #내부중첩함수명 내맘
    def innerfunction():      #함수이름 wrapper일 필요x
        start = time.time()
        callback()
        end = time.time()
        print(f"{callback.__name__} 실행시간:{end-start}초")
        #print(f"{callback.__name__} 실행시간: {end - start:.4f}초")
    return innerfunction #():없이 참조만 반환해야함

@time_decorator
def sigma( ):  
    s = 0
    for i in range(1, 10000000):
        s+= i
    print("합계: ", s) #출력: 합계:  49999995000000

sigma()  # 출력: sigma 실행시간:0.844681978225708초

print("ˇ"*40)
#0512_5시pm
#매개변수있고 반환값있는경우 데코레이터 만들기ㅣ
#callback - 뒤에서 호출한다
    #함수는 내가 만드는데 호출자는 시스템일때 callback
    #데코레이터한테 사용자함수 전달
#기본문법만 쓴 함수예제
def mydecorator(callback): #내맘대로 callback함수만듬
    def wrapper(*args, **kwargs): 
        result = callback(*args, **kwargs) #원래함수 호출하고
        return result
    return wrapper

@mydecorator
def add(x, y):
    return x+y

print( add(5,7)) # 출력 12

#def wrapper(*args, **kwargs): 데코레이터에서 가장 유연하고 강력한 형태의 함수방식
# *args 함수 :위치인자들을 튜플로 받음(1,2,3 ...)
# **kwargs 함수 : 키워드인자들을 딕셔너리로 받음(x=1, y=2 ...)
#*args(arguments인자들), kwargs(keyword 인자들)
#왜 이렇게 쓰냐면?
# 어떤 함수는 인자가 있을 수도, 없을 수도 있어요.
# *args, **kwargs를 쓰면 모든 경우를 커버.

print("ˇ"*40)
"""
문제 - 로그내보내기 데코레이터 만들기 문제
@mylog
함수를 sigma 매개변수 s = sigma2(10)

[LOG] 함수이름 : sigma2
[LOG] 입력값: args = (10), kwargs={}
[LOG] 반환값: 55
"""
#1.mylog먼저 만들기
def mylog(callback): #내맘대로 callback함수만듬
    def wrapper(*args, **kwargs): 
        result = callback(*args, **kwargs)#원래함수 호출하고
        print(f"[LOG] 함수이름 :{callback.__name__}")
        print(f"[LOG] 입력값 : arg = {args}, kwargs ={kwargs}")
        print(f"[LOG] 반환값 :{result}") 
        return result #반드시반환해줘야 전달됨
    return wrapper

#2.함수만들기
@mylog
def sigma2(limit=10):
    s = 0
    for i in range(1, limit+1):
        s+=i
    return s

#3.출력하기
print( sigma2(100))
print( sigma2(10) )

@mylog
def sub(a, b):
    return a-b
print( sub(3, 4))

print("ˇ"*40)""" 
개념	         자주 사용?	  중요한 이유
self	         자주 사용	인스턴스 접근 시 필수
@staticmethod	가끔 사용	인스턴스 없이도 호출할 때
@classmethod	가끔 사용	클래스 상태를 조작할 때 필요
*args, **kwargs	자주 사용	함수 매개변수 유연하게 받을 때
데코레이터	중간~가끔	코드 재사용, 로그, 보안, 실행시간 측정 등에 매우 유용

초보자는 자주 안 쓰지만, 실무·프로젝트에서는 꼭 필요합니다.
특히 데코레이터는 웹 개발, 머신러닝, 자동화에 자주 사용돼요.
"""
print("ˇ"*40)
#클로저(Closure) = 내부함수
#함수내 중첩함수 정의하고 내부함수가 외부함수의 지역변수에 접근가능할때,
#내부함수를 클로저라고 함.
    #함수저장하고 나중에 쓸때, 상태유지한채 함수실행하고 싶을때
def outer(msg): #외부함수
    def inner(): #내부함수(클로저)
        print("메시지:", msg)
    return inner #내부함수 자체를 리턴
myfunc = outer("안녕!") #outer실행 → inner리턴
myfunc()                #inner() 실행 → "안녕!"출력
