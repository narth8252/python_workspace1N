"""
0604 pm1:40 이중링크드리스트 = 바이너리 어쩌고랑 비슷
단일링크드리스트 뒤로추적은가능, 앞으로 추적은 불가능하다.
앞뒤로 추적할수있게 preview와 next 2개의 링크를 갖고있다.

 이중 링크드 리스트란 무엇일까요?
길에 신호등이 있다고 해요. 앞으로 가거나 뒤로 가는 길도 다 보여주는 것을 상상해 보세요.

이중 링크드 리스트는 바로 그와 같아요. 노드 노드라는 작은 상자들이 줄지어 있는데,
각 상자에는 값(데이터) 와 함께,
"앞에 있는 노드"를 가리키는 화살표 와 "뒤에 있는 노드"를 가리키는 화살표가 있어요.

그래서 우리가 길을 걷듯이, 이 노드들이 서로 연결되어 있어서,
앞으로 가거나 뒤로 가거나 답답하지 않게 찾을 수 있어요.

 왜 이렇게 할까요?
-앞으로 가기만 할 수 있는 리스트는요, 뒤로 가기가 어려워요.
-그런데, 이중 링크드 리스트는 앞으로도, 뒤로도 갈 수 있어서 아주 편리해요!
그림으로 쉽게 설명
처음 노드 (스티커 또는 그림 상자)에는 "시작"이라고 적혀 있어요.
그 다음 노드에는 "A", 그다음은 "B", "C", 이렇게 줄을 서서 연결되어 있어요.
각 노드에는 화살표가 두 개 있어서, 하나는 앞으로 가리키고, 또 하나는 뒤로 가리켜요.
예를 들어볼게요
우리가 노드를 만들어서 "C"라는 글자가 적힌 노드를 제일 앞으로 넣고 싶다면,
"C"는 "B" 앞에 넣어요.
그리고 "A"라는 노드도 뒤로 넣어요.
그 다음엔,

"C"의 뒤에는 "B"가 있고,
"B"의 뒤에는 "A"가 있어요.
이렇게 노드들이 연결되어 있어서, 우리가 이렇게 할 수 있죠:

앞에서부터 읽기 ("E", "D", "C", "B", "A")
뒤에서부터 읽기 ("A", "B", "C", "D", "E")
"""
# Node 클래스는 __init__() 안에 self.data=None이 중복되지 않도록,
class Node:
    def __init__(self, data=None):
        self.data = data
        self.data=None #앞의 노드주소간직
        self.next=None #뒤의 노드주소간직

class DoubleLinkedList:
    def __init__(self):
        self.head = Node() #가상head노드:리스트 전체의 시작 가리킴
        self.tail = Node() #가상tail노드:리스트 전체의 끝을 가리킴
        self.head.next = self.tail
        # self.tail.next = self.tail
        # self.head.prev = self.head
        self.tail.prev = self.head
        # 여기서, head와 tail은 더미 노드(실제 데이터는 이게 아닌 위치에 저장)
        #head <-> (|) <-> (|) <-> tail 주고받는것임

    def insertHead(self, data):
        temp = Node(data)
        # 새 노드(temp)를 헤드 다음에 연결
        temp.next = self.head.next
        temp.prev = self.head
        self.head.next.prev = temp
        self.head.next = temp

    def insertTail(self, data):
        temp = Node(data)
        # 새 노드(temp)를 꼬리 이전에 연결
        temp.prev = self.tail.prev
        temp.next = self.tail
        self.tail.prev.next = temp
        self.tail.prev = temp

    def printNext(self):
        t = self.head.next
        while t!=self.tail:
            print(f"{t.data}", end='')
            t = t.next

    def printPrev(self):
        t = self.tail.prev
        while t!=self.head:
            print(f"{t.data}", end='')
            t = t.prev
        print() #end써서 print넣어줘야 줄바꿈됨


    #삭제
    def deletHead(self):
        if self.head.next == self.tail:
            return
        self.head.next = self.head.next.next
        self.head.next.prev = self.head

    #삭제
    def deletTail(self):
        if self.tail.prev == self.head:
            return
        self.tail.prev = self.tail.prev.prev
        self.tail.prev.prev = self.tail

    def displayForward(self):
        current = self.head.next
        while current != self.tail:
            print(current.data, end=' ')
            current = current.next
        print()

    def displayBackward(self):
        current = self.tail.prev
        while current != self.head:
            print(current.data, end=' ')
            current = current.prev
        print()

#사용 예제
# 인스턴스 생성시에는 DoubleLinkedList()와 같이 반드시 괄호
dlist = DoubleLinkedList()
dlist.insertHead("A")
dlist.insertHead("B")
dlist.insertHead("C")
dlist.insertHead("D")
dlist.insertHead("E")
dlist.insertHead("F")

dlist.insertTail("G")
dlist.insertTail("H")
dlist.deletHead()
dlist.deletTail()

dlist.printNext()
dlist.printPrev()
    


