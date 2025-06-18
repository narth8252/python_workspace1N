#0513 10:40am 점프투파이썬 05-2모듈
#모듈안에 함수랑 클래스 만들자

#모듈 파일 생성
#module.py 파일을 생성하고, 아래의 코드를 입력합니다.
def add(x, y):
    return x+y

def sub(x, y):
    return x-y

class Person:
    def __init__(self, name, age=0):
        self.name = name
        self.age = age

    def print(self):
        print(f"name={self.name} age={self.age}")

print("--------구분:module.py 프린트---------------")
print( add(3,4))
print( sub(3,4))

print("--------구분:module.py 프린트---------------")
# print("모듈명",__name__)
if __name__ == "__main__":
    # print("이 파일을 직접 실행할 때만 실행됩니다.")
    print( add(3,4))
    print( sub(3,4))
    p1 = Person("조승연", 30)
    p1.print()
