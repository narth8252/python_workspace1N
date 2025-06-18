#0529 aobj0:30 예제
class Mytype:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def add(self):
        return self.x + self.y
        
    def sub(self):
        return self.x - self.y
        
    def mul(self):
        return self.x * self.y
    
#문제1. inspect써서 변수리스트 호출
#2. 함수리스트 호출
#3. setattr써서 x에는 , y에는5
#4. getattr써서 함수주소 호출

# 인스턴스 생성 빈칸으로
obj = Mytype()

#fields 정의
fields = [x for x in dir(obj) if not x.startswith("__")]
print(fields)

for field in fields:
    print(getattr(obj, field))

#인스펙트
import inspect

for field in fields:
    print( getattr(obj, field))

# 변수(필드)만 추출하는 컴프리헨션
var_fields = [name for name, value in inspect.getmembers(obj)
          if not (inspect.ismethod(value) or inspect.isfunction(value)) 
          and not name.startswith("__")]
print(var_fields)

#문제1. inspect써서 변수리스트 호출
for field in var_fields:
    print( getattr(obj, field))

#특정객체內 모든메서드와 변수들을 다 가져온다.
print( inspect.getmembers(obj))
for item, value in inspect.getmembers(obj):
    if inspect.ismethod(value) or inspect.isfunction(value):
        print("함수", item)
    else:
        print("변수", item)

#2. 함수리스트 호출

# 인스턴스에서 변수(필드)만 추출
var_fields = [name for name, value in inspect.getmembers(obj)
              if not (inspect.ismethod(value) or inspect.isfunction(value)) 
              and not name.startswith("__")]

print("변수 목록:", var_fields)

# 각 변수의 이름과 값 출력
for var in var_fields:
    print(f"{var} = {getattr(obj, var)}")

#3. setattr써서 x에는 10, y에는5
setattr(obj, 'x', 10)
setattr(obj, 'y', 5)

print(obj.x, obj.y)

#4. getattr써서 함수주소 호출


a = getattr(obj, 'x') #특정객체로부터 속성가져올수있다
print(a)

a = getattr(obj, 'y') #특정객체로부터 속성가져올수있다
print(a)

print(dir(obj)) #p클래스 내부에 있는 구조

#쌤풀이
fun_fields = [name for name, value in inspect.getmembers(obj) 
              if  (inspect.ismethod(value) 
                      or inspect.isfunction(value))
                and not name.startswith("__")]
print(fun_fields)
setattr(obj, var_fields[0], 10)
setattr(obj, var_fields[1], 5)
print(obj.x, obj.y)
print( getattr(obj, fun_fields[0])())
print( getattr(obj, fun_fields[1])())
print( getattr(obj, fun_fields[2])())

