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