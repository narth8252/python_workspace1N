# 0602 pm1시 자료구조-배열과 연결리스트
# https://wikidocs.net/224937
# 좌충우돌, 파이썬으로 자료구조 구현하기 03장 파이썬으로 연결 리스트 구현하기

# 자료구조 정리: 배열과 연결 리스트
# 1. 배열(자료구조)로 데이터 삽입 과정을 이해하기
#배열끼워넣기
arr = list()
# 배열에 10개 미리 데이터 넣기
for i in range(0,10):
    arr.append(i+1)

# 3번방에 88 넣기(기존3번이후 데이터 한칸씩 뒤로밀기)
for i in range(9,3,-1):
    arr[i] = arr[i-1]
arr[3] = 88
print(arr)
#단점: 배열은 중간에 데이터 삽입시, 뒤데이터들이 한칸씩 밀리기 때문에 Overhead(시간/메모리 낭비)가 발생한다.


#클래스나 구조체를 써야한다
#배열(Array) 기반 자료구조의 단점을 보여준 후, 그 대안으로 연결 리스트(Linked List) 를 직접 구현하려는 흐름이야.
# head -> [노|드] -> [노|드] -> tail
# head[100] -> [  |200]
# tail[200] -> [  |200]
# -------------------------
# temp[300] -> ["A"|  ] #"A"옆노드에 200이 오고 temp는 지역변수니까 사라짐.
#                    ↖temp=node▶200

#250602 pm2:30 링크드리스트(리스트1.py)
# 2. 연결 리스트(Linked List) 구현
# 2-1. 노드(Node) 구조체/클래스

# Node / MyList 클래스 기본 구현
class Node:
    def __init__(self, data=None):
        self.data = data  # 데이터 저장
        self.next = None    # 다음 노드 참조

# 2-2. 연결 리스트(MyList) 구조
class MyList:
    #head와 tail 리스트의 시작과 끝
    #head와 tail은 지워지면안돼서 생성자에서 해라
    def __init__(self):
        self.head = Node() #더미 head sentinel 노드:안씀.자동차범퍼역할:중간에 데이터언제나 끼워넣기위해
        self.tail = Node() #더미 tail sentinel 노드:안씀.자동차범퍼역할:중간에 데이터언제나 끼워넣기위해
        self.head.next = self.tail
        self.tail.next = self.tail #None의미임. 근데 코딩이 tail로 써야 편함,tail은 끝 노드
        # head -> (|) -> tail(|) -> None
        # 이 상태에서는 아직 연결된 실제 데이터는 없음. 이후 insert 메서드를 만들면 진짜 쓸 수 있어.

    # 1. 머리(앞쪽) 삽입
    def insertHead(self, data):
        temp = Node(data)
        temp.next = self.head.next
        self.head.next = temp

    # 2. 리스트 내용 출력 (단순 예시)
    def print(self):
        print(self.head.next.data)
        print(self.head.next.next.data)
        print(self.head.next.next.next.data)
    
    # 3. 리스트 전체 출력 (while 이용)
    def print2(self): #head, tail값은 불변, 추정용Node타입만선언해서 만든다기보다 선언해서 쫓아감
        trace = self.head.next
        while trace != self.tail:
        #trace가 tail이 아닐동안반복
        # trace = trace.next
        # print(trace.data)
            print(trace.data, end=' -> ')
            trace = trace.next
        print("끝")

    # 4. 머리 삭제 (헤드 삭제)(delete) – Singly Linked List
    def deleteHead(self):
        if self.head.next == self.tail:
            return  # 비어있으면 종료(이미 다 삭제돼서 없음)
        self.head.next = self.head.next.next
    # prev = self.head
    # current = self.head.next

    # while current != self.tail:
    #     if current.data == target_data:
    #         prev.next = current.next  # current 제거
    #         return True
    #     prev = current
    #     current = current.next
    # return False  # 못 찾음

    # 5. 전체 삭제
    def deleteAll(self):
        self.head.next = self.tail

# 2-3. 사용 예제
# ml = MyList()
# ml.insertHead("A")
# ml.insertHead("B")
# ml.insertHead("C")
# ml.print2()  # C -> B -> A -> 끝

# print("삭제 후")
# ml.deleteHead()
# ml.print2()  # B -> A -> 끝
"""
3. 자료구조에 대한 핵심 개념 정리
Node	  : 데이터와 다음 노드를 가리키는 구조체 또는 클래스
head, tail:	시작과 끝의 더미 노드 역할, 데이터 노드와는 별개로 존재할 수 있음
삽입 방법  : 이전 노드(prev_node)의 next를 새 노드로 연결하는 방식
장점	:배열과 달리 중간 삽입 삭제 시 기존 데이터 이동이 필요 없음 (Overhead 적음)
단점	:뒤쪽 순서만 쉽게 탐색 가능, 앞쪽 탐색은 어렵거나 불가
4. 주의
단일 연결 리스트(Singly Linked List)는 앞쪽에서 뒤쪽으로만 연결되어 있기 때문에, 뒤쪽 노드 이동이 불가능한 단점이 있음.
이중 연결 리스트(Doubly Linked List)는 앞뒤 두 방향의 포인터를 가지고 있어 양방향 탐색이 가능.
환형 연결 리스트(Circular Linked List)는 리스트 끝과 처음이 연결되어 원형 구조를 형성.
"""
    # 6. 끼워넣기(insert) flag사용
# insertOrder 함수:정렬된 연결 리스트에 새 데이터를 적절한 위치에 삽입하여 리스트를 항상 정렬 상태로 유지
# 구조
#  -temp: 새로 삽입할 노드
#  -t1, t2: 각각 현재 탐색 중인 노드, 그 바로 앞 노드
#  -flag: 위치를 찾았는지 여부를 저장하는 변수

def insertOrder(self, data): 
    temp = Node(data)  # 새 노드 생성
    t1 = self.head.next  # 현재 노드 (처음 노드)
    t2 = self.head       # 이전 노드 (더미(head),뒤에서 따라감
    flag = False  #while문 종료여부를 위한 flag 변수

    #위치찾기: t1이 tail이 아니고, t1의 데이터가 새데이터보다 크지않으면 계속반복
    #위치를 찾거나 마지막에 도달했을때 끝내야함
    #not flag: flag값이 False인동안 반복
    while not flag and t1!=self.tail:
        if t1.data > data:
            flag = True  #적절한 위치를 찾았을 때 종료 조건 설정
        else:
            t1 = t1.next
            t2 = t2.next

            flag = False

            while not flag and t1.next!=self.tail:
                if t1.data > temp.data:
                    flag = True
                else:
                    t1 = t1.next
                    t2 = t2.next
    #t2와 t1사이에 temp를 끼워넣는다
    #insertHead와 insertOrder를 같이 쓰면안됨.
    temp.next = t1
    t2.next = temp

    ##4.검색 search
    # def searchHead(self):
    # current = self.head.next
    # while current != self.tail:
    #     if current.data == target_data:
    #         return current
    #     current = current.next
    # return None



m1.insertOrder("A")
m1.insertOrder("B")
m1.insertOrder("C")
m1.insertOrder("D")
m1.print2()


# | 개념             | 설명                                      |
# | -------------- | --------------------------------------- |
# | `Node`         | 데이터 하나를 담고, 다음 노드를 가리키는 구조체             |
# | `head`, `tail` | 더미 노드 (데이터 없는 시작/끝 노드)                  |
# | 삽입             | `prev_node.next` 에 새로운 노드를 끼워넣는 식       |
# | 장점             | 배열처럼 메모리 밀어넣기 없이 중간 삽입 가능 (overhead 없음) |

#단일링크드리스트의 단점: 뒤의순서는 가능하나 앞순서는 추적불가능
#중간에 링크가 끊어지면 그걸로 끝
#이중링크드리스트 - prev(앞), next(뒤) 2개의 링크를 갖는다
#환형링크드리스트(한바퀴도는것) 안씀: 책에 있으면 배울필요X 
#큐라는 구조를 만들때 배열의 경우
# <------ [ , , , , , , ] <-------

#본래의 링크드리스트는 데이터가 정렬되어서 들어가야 한다.
""" 그림
   head ->
   1. head에 아무것도 없을때
   2. 리스트의 맨 끝에 데이터가 추가될때
   3. 중간에 끼워넣을때
    자료구조 책 세화출판사 이재규 심플하게 만듬

"""

# 예시
# 예시
class Book:
    def __init__(self, title="", author="", publisher=""):
        self.title = title
        self.author = author
        self.publisher = publisher
    def __gt__(self, other):  # > 연산자중복, Book 클래스의 __gt__(greater than) 연산자 오버로딩
        return self.title > other.title  # 간결하게

    def __str__(self):
        return f"{self.title} {self.author} {self.publisher}"
    
# def myfunc(o):
#     return o.title
    
print( Book("마법사의돌", "조앤롤링", "해냄") > 
       Book("마법사의돌", "조앤롤링", "해냄" ))
print( Book("마법사의돌", "조앤롤링", "해냄") > 
       Book("그리고아무도없었다", "아가사크리스티", "해냄") )

m2 = MyList()
m2.insertOrder( Book("마법사의돌", "조앤롤링", "해냄") )
m2.insertOrder( Book("쌍갑포차", "배혜수", "카카오") )
m2.insertOrder( Book("뽀짜툰", "유리", "카카오") )
m2.insertOrder( Book("무빙", "강풀", "카카오") )
m2.print2()

b1 = Book("쌍갑포차")
b2 = Book("쌍갑포차")
print( b1 == b2 ) #b1과 b2내용비교가 아니고, b1이 참조만저장, b2도 참조만저장
s1=str("Hello")
s2=str("Hello")
print( s1 == s2)
#b1이 참조만저장, b2도 참조만저장
#둘이 동일한 객체를 참조하고있는가?
#"Hello".equals(temp)