#0529 am9, 10시
#이런식의 코딩도 가능하다고 한번 해보는 것임. 몰라도됨. 가볍게 들어라, 그런갑다.
#데코레이터 만들때 함수추출해서 쓸때, 어떤게 매개변수로 올지모를때
#쌤도 프레임워크 갖다쓰지 만들진않음.

# 리플렉션 - 거울
# 클래스를 만들면, 언어 번역가들이 클래스를 읽어서 정보를 해석해서 저장
# 그 정보를 접근할 수 있게 해준다.
# 실행 중인 객체나 클래스의 구조를 조사하고 조작하는 기능. 
# 보통 Java나 C#에서 많이 언급되지만, Python도 매우 강력한 리플렉션 기능을 가지고 있어.
# 프레임워크 만들때 사용자가 클래스를 설계함
# a = 클래스명

class Person:
    def __init__(self, name="", age=20):
        self.name = name #2개의 필드가 있다.
        self.age = age
        
    def greet(self):
        print(f"Hello {self.name}")

p = Person("Tom", 12)

#클래스內 속성가져온다
a = getattr(p, 'name') #특정객체로부터 속성가져올수있다
print(a)

a = getattr(p, 'age') #특정객체로부터 속성가져올수있다
print(a)

print(dir(p)) #p클래스 내부에 있는 구조
#필터링을 해서 사용자가 만든거만 
fields = [x for x in dir(p) if not x.startswith("__")]
print(fields)

for field in fields:
    print(getattr(t, field))

import inspect #이 라이브러리가 각요소가 함수인지 아닌지 확인가능
for field in fields:
    print( getattr(p, field))

#특정객체內 모든메서드와 변수들을 다 가져온다.
print( inspect.getmembers(p))
for item, value in inspect.getmembers(p):
    if inspect.ismethod(value) or inspect.isfunction(value):
        print("함수", item)
    else:
        print("변수", item)

#[출력변수 for 출력변수 in iterabletype if 조건식]

#변수(필드)만 추출하는 리스트 컴프리헨션임(클래스로부터 변수이름만 알아내기)
#꼭이렇게 안해도됨. for루프 다른방법多
var_fields = [ name for name, value in inspect.getmembers(p)
               if not (inspect.ismethod(value) or inspect.isfunction(value))
               and not name.startswith("__")]
print(var_fields)


#클래스로부터 함수,method이름만 알아내기) not빼
var_fields = [ name for name, value in inspect.getmembers(p)
               if (inspect.ismethod(value) or inspect.isfunction(value))
               and not name.startswith("__")]
print(var_fields)

getattr(p, var_fields[0]) #0번째위치에 뭐있는지모르지만 가져와봄
print( a )

#객체內 변수값바꾸는 설정도 가능
setattr(p, 'name', '홍길동')
setattr(p, 'age', 43)

print(p.name, p.age)

#메소드 가져와서 호출가능
def add(x,y): #함수도주소라 변수들에 주소저장가능 → 변수통해호출가능
    return x+y
a = add
print( a(5,6))

#얘네입장에서는 변수,함수 차이x
method = getattr(p, "greet")
method()

#함수의 매개변수도 알수있다
params = inspect.signature(add)
print(params)