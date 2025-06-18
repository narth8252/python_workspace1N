# 0602 pm4:20 스택 https://wikidocs.net/192069  좌충우돌,파이썬으로 자료구조 > 04장 파이썬으로스택구현하기
#시험문제에 답 스택 많이나옴
#실무에서 우리가 만들어쓰지X, 내부구조보려고 이해도높이려고 만들어봄.

# 스택에서의 "코딩적 의미"
# 1. 스택의 기본 개념
# 스택(Stack)은 후입선출(LIFO, Last-In-First-Out) 구조를 갖는 자료구조입니다.
# 데이터를 넣을 때(푸시), 넣은 순서의 반대로 꺼낼 때(팝) 처리됩니다.
# 주로 호출 스택(call stack)**이라는 개념으로 프로그래밍에서 매우 중요하게 사용됩니다.

# 2. 프로그래밍에서 스택의 역할
# 함수 호출과 반환:
#  함수를 호출하면, 그 함수의 실행 상태(지역 변수, 명령 위치 등)가 호출 스택에 저장됩니다.
#  함수가 종료되면, 이전 상태를 복원하는데 스택에서 값을 꺼내어 돌아옵니다.
# 재귀적 호출:
#  재귀 함수는 반복적으로 자기 자신을 호출하며, 매 호출 시마다 호출 정보(지역 변수, 반환 주소 등)가 스택에 쌓입니다.
#  종료 조건이 충족되면, 스택에 쌓인 호출 정보들이 차례로 제거(반환)됩니다.

# 3. 코딩적 의미
# 구현의 핵심:
#  재귀 함수 또는 중첩 호출 시, 호출 시점의 위치, 지역 변수, 반환 주소 등을 저장하는 공간 역할.
#  그래서 "코딩적 의미"는, 호출 스택이 함수의 호출과 종료, 즉 프로그램 흐름의 제어를 가능하게 하는 구조적인 역할을 의미합니다.
# 효율성과 제한:
#  재귀 호출이 깊어지면 스택이 가득 차서 StackOverflow 에러가 발생, 즉 스택 크기 한계를 의미하기도 함.
#  이는 프로그래머에게 "함수 호출의 깊이와 종료 조건 설계"라는 프로그래밍 원리와 직결됩니다.

"""
스택 - Last In First Out - LIFO 후입선출구조
인터럽트 - 가로채기, 컴퓨터는 한번에 하나만 한다.
읽고 계산하고 출력하고(사이클)
컴퓨터CPU는 제할일 계속하고있으면 우리가 입력할게 있으면 인터럽트 검
출력할게 있으면 인터럽트 검 -> 이때 대표적으로 
함수 - 스택
수식트리, 트리순회, 0주소
대기줄, 줄서는거 - 큐구조
연산: push, pop, peek, isFull, isEmpty
push - 스택의 마지막에 데이터넣기
pop - 스택의 마지막에 하나 꺼내서 반환하기
peek - 스택의 마지막 데이터 반환만 한다
isFull - 스택이 꽉차면 True, 아니면 False반환
isEmpty - 스택이 비면 True,  아니면False반환
이 구조가 뭔지 안중요함. 저런 연산이 있기만 하면 됨.
내부 데이터를 배열로 둔다
[0,0,0,0,0,0,0]       0 1 2 3 4 5 6 7 8 9
top 인덱스 필요 - 스택의 마지막 데이터를 가리킨다.
"""
#객체지향의 장점:스택이 많이 필요한데 객체몇개만 만들면됨.
class MyStack:
    def __init__(self, size=10):
        if size<10:
            self.size=10 #최소크기를 10으로 하자
        else:
            self.size = size
        self.stack=[]
        for i in range(0, self.size):
            self.stack.append(0) #배열이없어서0으로 잡는수밖에.
        self.top = -1

    # push 함수 만들기: 수도코드해라
    #데이터가 isFull상태가 아니면 top증가시키고 그안에 값넣기
    def isFull(self):
        if self.size-1 == self.top:
            return True
        return False
    
    def push(self, data):
        if self.isFull():
            return
        self.top += 1
        self.stack[self.top]=data
    
    def print(self):
        i=0
        while i <=self.top:
            print(self.stack[i], end=" ")
            i+=1
        print()

    #비어있는지 여부 판단하는 isEmpty() 함수
    def isEmpty(self):
        # return self.top == -1 #1줄로 끝내기
        if self.top == -1:
            return True
        return False
    
    def peek(self):
        if self.isEmpty():
            return None
        return self.stack[self.top]
    
    def pop(self):
        if self.isEmpty():
            print("스택이 비어있습니다.")
            return None
        item = self.stack[self.top]
        self.top -= 1
        return item
    
if __name__=="__main__":
    s1 = MyStack()
    s1.push('A')
    s1.push('B')
    s1.push('C')
    s1.push('D')
    s1.push('E')
    s1.push('F')
    s1.push('G')
    s1.push('H')
    s1.push('I')
    s1.push('J')
    s1.push('K') #stack이 10개 full이라 안들어감
    s1.push('L')
    s1.print() #A B C D E F G H I J 
    print("--------------------------")

    print( s1.pop()) #J
    print( s1.pop()) #I
    print( s1.pop()) #H
    print( s1.pop()) #G
    print( s1.pop()) #F
    print( s1.pop()) #E
    print( s1.pop()) #D
    print( s1.pop()) #C
    print( s1.pop()) #B
    print( s1.pop()) #A
    print( s1.pop()) #스택이 비어있습니다. None

#스택 사용해서 문자열 뒤집기
# s2 = MyStack(30)

def reverse(arr):
    s = MyStack(len(arr))
    for i in arr:
        s.push(i)
        
    result =""
    while not s.isEmpty():
        result += s.pop()
    return result
print( reverse("Korea")) #aeroK

# 문자열 뒤집기 함수
def reverse_string(s):
    s2 = MyStack(len(s))
    
    # 문자열의 각 문자들을 스택에 넣기
    for char in s:
        s2.push(char)
    
    # 스택에서 문자들을 꺼내서 뒤집기
    reversed_str = ""
    while s2.top != -1:  # 스택이 비어있지 않을 동안
        reversed_str += s2.pop()
    
    return reversed_str

# 사용 예시
original = "안녕하세요"
reversed_str = reverse_string(original)
print("원래 문자열:", original)       #원래 문자열: 안녕하세요
print("뒤집은 문자열:", reversed_str) #뒤집은 문자열: 요세하녕안