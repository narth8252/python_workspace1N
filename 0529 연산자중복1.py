#0529  PM 1:40

#연산자중복 - 이미정해져있음. 이름,반환형태,매개변수 정해져있다.
# m3 = m1 + m2
# m3 = m1.__add__(m2)
# result = m2 + m1
# m2.__add__(m1)
# self - class內 메소드들은 객체자신에 대한 참조로 누구나 self가져야함
# other는 전달받은 매개변수
# 반환값이 객체여야함
class MyType:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return MyType(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return MyType(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return MyType(self.x * other.x, self.y * other.y)

    def __str__(self):
        return f"MyType(x={self.x}, y={self.y})"

# 테스트 코드
m1 = MyType(4, 5)
m2 = MyType(8, 9)

m3 = m1 + m2
print(m3)                  # MyType(x=12, y=14)
print(m1 - m2)             # MyType(x=-4, y=-4)
print((m1 - m2).__str__()) # MyType(x=-4, y=-4)

