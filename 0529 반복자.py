#0529 am11시
""" 
반복자: 데이터를 저장하는 구조
1. 선형구조 - 배열과 링크드리스트
1-1. 배열(array)
연속된 메모리공간이 필요
주기억장치(RAM)에 연속된 공간이 없으면 할당 못함
예)100mb 필요하면 100mb줄수있어야한다
   10,20,,40,30조각내서 단편화
   OS가 주기억장치RAM을 압축해서 100mb를 만들어서 OS가 배당(메모리단편화)
   메모리 단편화(Fragmentation)
        - 외부 단편화 (External Fragmentation)
        메모리 공간이 여러 조각으로 흩어져 있어서, 총합은 충분하지만 연속된 공간이 없음.
        예: 10MB + 20MB + 30MB + 40MB = 100MB 있음 → 그러나 연속 공간이 아님 → 100MB 할당 못 함.
        - 내부 단편화 (Internal Fragmentation)
        할당된 메모리 블록 내부에서 실제 사용되지 않고 낭비되는 공간.
        예: 100MB 할당했지만 실제로는 85MB만 사용 → 15MB 낭비됨.
        - 해결책 : 메모리 압축 (Compaction)
        OS가 메모리를 재배치하여 연속된 공간을 확보함.
        사용 중인 메모리 블록을 앞쪽으로 당겨서 빈 공간들을 모음.
        압축 과정은 CPU 자원을 많이 사용하고, 성능에 영향을 줄 수 있음.
        일반적으로는 OS에서 자동으로 실행되거나, 특정 시점에 수동 트리거됨.

데이터 접근시 순서대로 접근하는 Index 사용
본래의 배열은 프로그램 시작전에 메모리크기를 정확하게 지정해서 옮겨다닐수없었고 변경불가
a = []        a[0] = 10   print(a[1])
a.append(1)
a.append(2)
현재의 배열은 index를 사용한다는 공통점 하나만 남음
배열의 장점은 첫시작위치를 알면 다음요소가 바로옆(index순차적)에 있어서 주르륵 읽으면 됨
그래서 속도빠름
단점 : 융통성이 없어서 필요시 메모리RAM을 늘리거나 줄일수없다. 그래서 미리 크게확보.
데이터 중간에 끼워넣으면 index순차적이므로 그 뒤데이터들이 주르륵 뒤로 밀리며 이동해야함
그래서 중간 삭제,삽입이 어려움=오버헤드 발생.
근데 컴퓨터가 일을 많이하는거지, 개발자는 편할수있음.간단하고 심플코딩.

1-2. 링크드리스트(Linked List)
데이터와 다음번요소에 대한 주소 저장
목걸이공예 비즈로 연결시켜나가는 개념
데이터+주소
(데이터ㅣ주소) → (데이터ㅣ주소) → (데이터ㅣ주소)
장점: 필요한만큼만 만들어서 사용가능,데이터 중간에 끼워넣거나 삭제 쉽다.
단점: 인덱스사용불가, 데이터접근 어려움, 느리다.
(파이썬의 LIST: 배열+링크드리스트 느낌, 
     실제로 class를 만들면 class내부에 배열을두고 이 내부배열을 접근하게하는 수단들 연산자중복)

링크드리스트: 메모리연속성없이 데이터연결해 저장하는 대표 자료구조. 
지금까지 얘기한 메모리 단편화와도 관련이 있어, 
왜냐하면 연속된 공간이 없어도 노드를 어디든 저장할 수 있기 때문이지. 
아래에 개념, 구조, 장단점, 그리고 간단한 예시까지 정리해줄게.

 링크드 리스트(Linked List)란?
데이터를 노드(Node) 단위로 저장하고,
각 노드는 자신의 데이터 + 다음 노드에 대한 참조(pointer) 를 가지고 있음.

배열(array)과 달리 메모리 연속성이 필요없음.

 구조
1. 단일 연결 리스트(Singly Linked List)
[Data|Next] → [Data|Next] → [Data|Next] → None
각 노드는 데이터와 다음 노드 주소를 가짐.

2. 이중 연결 리스트(Doubly Linked List)
None ← [Prev|Data|Next] ⇄ [Prev|Data|Next] ⇄ [Prev|Data|Next] → None
양방향으로 연결 → 앞뒤 탐색이 가능.

3. 원형 연결 리스트(Circular Linked List)
[Data|Next] → [Data|Next] → [Data|Next] ┐
       ↑                                │
       └────────────────────────────────┘
마지막 노드가 첫 번째 노드를 가리킴 → 순환 구조.

 장점
 메모리조각 많은 환경에 유리.리스트에 중간삽입/삭제가 빈번할때.
큐(Queue), 스택(Stack), 해시맵(HashMap)의 일부 구현 방식으로도 사용됨.
메모리가 조각나 있어도 문제 없음.
노드 삽입/삭제가 빠름 (중간에 끼워넣기 쉬움).
크기 변경이 자유로움 (동적 구조).

 단점
랜덤 접근 불가 → 인덱스로 바로 접근할 수 없음 (O(n))
메모리를 더 씀 → 포인터도 저장해야 함.
순회 시 시간이 더 걸림.

 예제 (파이썬 - 단일 연결 리스트)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None  # 다음 노드 주소

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = new_node

    def display(self):
        cur = self.head
        while cur:
            print(cur.data, end=" → ")
            cur = cur.next
        print("None")

# 사용 예시
ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.display()  # 출력: 10 → 20 → 30 → None

2. 비선형구조 - 트리, 그래프
부모와 자식으로 나뉜다
    a         트리구조를 순회하려면
  B   C       DFS(깊이우선탐색) - STACK
D   E   F     BFS(너비우선탐색) - 큐

그래프는 전체 망형태를 말한다

각각의 데이터유형에 따라 순회방법 다르다.
사용자에게 동일한 접근방법 제시 - 반복자(iterator)
컬렉션류 : list, dict, tuple, set등 그밖의 라이브러리들
내부데이터 접근방법은 통일iterator제공
"""
a = [10,20,30,40]
print(a[0])
it = iter(a) #반복자 iter라는 함수를 만들고 (iterator+연산어쩌고 합친기능)
print(next(it)) #얘한테 넘기면 반복자 차례대로 데이터호출가능
print(next(it))
for i in a: #원래 for구문으로 안됐는데 반복자가 가능하게 한게 위에 iter함수 만든것
    print(i)

b = {"red":"빨강", "green":"초록"} #dict타입으로 해보자. 예전엔 안됐음
it2 = iter(b) #반복자iter함수로 print가능하게 해줌
print( next(it2) ) #내부구조는 다른데 알면 나만의 것 만들수있음.
print( next(it2) )
print( next(it2) )
#반복자 목적: 사용자가 class내부  동일한 방법으로 접근가능하게
#컬렉션class설계자들이 공통의 인터페이스를 정의해놓고 구현한것

#반복자 가져오는 또다른방법
#__시작하는 함수들은 내장함수
it = a.__iter__()
print( next(it)) #반복자의 현재위치값을 반환하고 반복자가 다음번요소로 이동
#더이상 읽을 데이터가 없으면 파이썬은 StopIteration이라는 예외발생시킴.
#보통은 직접 이렇게 쓰지말고, while도 별로고 for문쓰면 알아서 할거다
for i in a:
    print(i) #for문쓰면 iterator가져가서 하게 개발자 편하게 하겠다.

it = a.__iter__()
while True:
    try:
        item = next(it)
        print(item)
    except StopIteration:
        print("이터레이터종료")
        break #while문종료

#이거가능하게 하고싶어서 반복자라는 개념사용
for i in a:
    print(i)

#반복자 이렇게까지 안하려고 했는데 클린코드책에 자꾸나와서
#실무에 만들필요없지만 해보자.
#반복자 만드려면 연산자중복부터 배워서 만들어야 한다.
#반복자 구축방법은 언어마다 다른데 인터페이스냐 연산자중복이냐라는 것
#인터페이스: 클래스인데 구현부분이 없dl 함수머리통만 있는 class

#설명하기 힘들어 살짝 미친짓인데 만들어보자
#오늘보고 잊어버려라.
class MyInterface:
    def __init__(self, add=None, sub=None):
        self.add = add
        self.sub = sub

 #상속
 #인터페이스는 실제구현부분이 없는 함수들의 묶음
 #그래서 객체생성이 안됨.
class MyCalculator(MyInterface):
    def __init__(self):
        # 부모 생성자 호출 + 구현 함수 등록
        super().__init__(add=self.add2, sub=self.sub2)

    def add2(self, x, y):
        return x + y
        
    def sub2(self, x, y):
        return x - y

# 사용 예시
obj = MyCalculator()
print(obj.add(10, 5))  # ➜ 15
print(obj.sub(10, 5))  # ➜ 5

    




