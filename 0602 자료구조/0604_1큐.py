"""
0604 am9시 https://wikidocs.net/192523 좌충우돌,파이썬으로자료구조05파이썬으로큐구현
[ 큐 ]
1. 선입선출(ex.은행의 대기줄)
2. 버퍼=큐: 데이터를 모았다가 한번에 처리하는 방식
    컴퓨터의 입출력장치와 메모리간의 속도차가 너무 커서 
    일부 메모리공간을 잘라서 우리가 키보드를 누르면 그값이 버퍼라는 공간에 먼저 들어갔다가
    엔터키를 누르면 그때 메모리로 들어간다.
3. 배열 - 환(원)형 큐 동그라미
4. 링크드리스트 - 더블링크드리스트
5. 우선순위큐 : 우선순위를 준다. 우선순위따라서 데이터 순서가 바뀐다.
6. 메시지큐: 윈도우OS안에 있음.
            사용자의 동작(이벤트)이 많이 발생하면 컴퓨터 감당안되니
            각각의 이벤트에 번호를 붙여서 어디서 무슨일이 있었는지 기록해서
            큐에 저장한다.
7. 양방향큐(데큐) : 필요에 의해서 양방향에서 데이터 넣고빼기 가능(쓸일X,용어만.)
8. 기본큐 : 한쪽방향에서 데이터를 넣기만하고 반대방향에서는 데이터를 가져가기만 하는 구조
큐에서는 head 대신 front라고 하며, 마지막에 들어간 노드 쪽을 rear (or back)이라 한다.
enqueue: 가장 마지막에 자료를 넣는 것으로 위의 그림을 보면 연결 리스트의 append와 같다.
dequeue: 가장 먼저 들어간 자료를 꺼내는 것으로 연결 리스트의 popleft와 비슷하다.
큐를 이용한 코드는 짧고 명확하지만 리스트인덱스가 더 빠르다.


front - 이쪽에서 데이터를 가져간다. 
rear(back) - 이쪽에서 데이터를 추가한다.

9. 큐의 주요연산 함수
 put() : 큐에 데이터 넣는 연산
 get() : 큐에서 데이터 가져오기 연산
 isFull() : 큐가full이면 True, 아니면False
 isEmpty() : 큐가Empty면 True, 아니면False
 peek() : 큐의 맨첫값 하나를 확인하는 용도

10. 그림만 기억(네모방/원형방)을 코드실행으로 표현하자면?
0 1 2 3 4
0 1 2 3 4
0 1 2 3 4
front =0  rear=0     front ==rear empty상황
pur('A')             0 'A'  0  0  0
front =0  rear=1
put('B')             0 'A' 'B' 0  0 
front =0  rear=2
put('C')             0 'A' 'B' 'C' 0 
front =0  rear=3
put('B')             0 'A' 'B' 'C' 'D'
front =0  rear=4
put('E')  (rear+1)%5===front  full
front =1  rear=4
get('A')             0  0 'B' 'C' 'D'
front =2  rear=4
get('B')             0  0  0 'C' 'D'
front =3  rear=4
get('C')             0  0  0  0 'D'
front =4  rear=4
front==rear Empty            0  0  0  0  0
"""

class MyQueue:
    def __init__(self, size=10):
        if size<10:
            self.size = 10
        else:
            self.size = size
        self.queue = [None] *self.size
        print(self.queue)
        self.front = 0 #데이터를 꺼내는 위치
        self.rear = 0 #데이터를 넣는 위치

    #원형 큐에서 Full 상태 확인
    #한 칸을 비워두어 full과 empty를 구분
    def isFull(self): #이게 한바퀴돌게 만드는 파트
        #rear가 마지막 위치이고 front가 0이거나 또는
        #rear+1(1칸씩옆으로 옮김)이 front와 같으면 Full
        if (self.rear+1)%self.size == self.front:
            return True
        return False
    
    def isEmpty(self):
        #front와 rear가 같은위치면 Empty
        if self.front == self.rear:
            return True
        return False
    
    def put(self, data):
        #Full이면 저장불가
        #rear+1칸씩 옆으로 옮기고 데이터개수self.size만큼 나누면 %self.size
        if self.isFull():
            print("Queue is full")
            return
        #rear위치에 데이터 저장하고 rear이동
        self.rear = (self.rear+1)%self.size
        self.queue[self.rear]=data

    def get(self):
        #Empty면 꺼낼 데이터 없음
        if self.isEmpty():
            print("Queue is Empty")
            return None
        #front(증가)이동하고 그위치의 데이터값 반환
        self.front = (self.front +1)%self.size
        return self.queue[self.front]
    
    def peek(self):
        #Empty면 확인할 데이터 없음
        if self.isEmpty():
            print("Queue is Empty")
            return None
        #front+1(증가)이동한 그위치의 데이터값 반환(꺼내지는 않음)
        #front 자체는 바꾸면 안됨
        # return self.queue[(self.front+1)%self.size]
        temp = (self.front+1)%self.size
        return self.queue[temp]

    def print(self):
        print("Queue상태보기")
        temp = self.front%self.size
        while temp != self.rear:
            temp = (temp+1)%self.size
            print(self.queue[temp], end="\t")
            # print("front", self.front, "rear", self.rear, temp)

# 테스트: 새로운 큐 객체를 생성
# A부터 J까지 10개의 데이터를 순차적으로 삽입
# 최종 큐의 상태를 출력
# 만약 큐의 크기가 10보다 작다면, Full 상태가 되어 모든 데이터가 들어가지 않을 수 있습니다. 
# 큐의 크기를 확인하시는 것이 좋습니다.
if __name__=="__main__": #쌍따옴표
    q = MyQueue()
    q.put('A')
    q.put('B')
    q.put('C')
    q.put('D')
    q.put('E')
    q.put('F')
    q.put('G')
    q.put('H')
    q.put('I')
    q.put('J')
    q.get()
    q.get()
    q.get()
    q.put('J')
    print(q.queue)

    print(q.peek())
    print(q.queue)

    while not q.isEmpty():
        print(q.get())

"""
#원형큐의 작동원리
# 1. 처음 상태
[A][B][C][D][E]  # 큐가 가득 찬 상태
 0  1  2  3  4
    f           r

# 2. B를 꺼내면 (get 연산)
[A][x][C][D][E]  # B가 나가고 빈자리가 생김
 0  1  2  3  4
       f        r

# 3. F를 넣으려고 할 때
# rear = (rear + 1) % size 연산으로 인해
# rear가 4에서 0으로 순환됨

# 4. F가 들어간 최종 상태
[F][x][C][D][E]  
 0  1  2  3  4
       f  
 r

F가 0번방에 들어가는 이유:

rear가 4(마지막 인덱스)에 있을 때
새로운 위치 = (rear + 1) % size
(4 + 1) % 5 = 0
따라서 새로운 rear는 0이 되어 F는 0번방에 저장
이것이 "원형" 큐인 이유입니다:

배열의 끝에 도달하면 다시 처음으로 돌아가서 저장
마치 원형으로 연결된 것처럼 동작
"""

""" 0604 pm 1시 문제풀이
은행에 가면 순번뽑기
1.은행원 - 작업완료 -> 몇번 손님 오세요
2.고객 - 번호뽑기
"""
class BankNumber:
    def __init__(self):
        self.number=0 #고객번호가 0 ~~~~~  
        self.cnt=0    #현재 대기 인원수 
        self.queue = MyQueue(500) #큐의 크기 500 

    def menuCustomer(self):
        self.number += 1
        self.cnt+=1 
        self.queue.put(self.number)
        print(f"대기인원 {self.cnt}" )
        print(f"고객님 번호는 {self.number} 입니다")
    
    def menuBanker(self):
        #큐에서 번호하나를 꺼내 고객있는지 확인
        if self.queue.isEmpty():
            print("대기중인 고객이 없습니다.")
            return 
        number = self.queue.get() 
        print(f"{number}번 고객님 창구앞으로 와주세요")
        self.cnt -= 1
        print(f"대기인원 {self.cnt}" )

    def main(self):
        while True:
            sel = input("1.고객 2.은행원 3.종료")
            if sel=="1":
                self.menuCustomer()
            elif sel=="2":
                self.menuBanker()
            elif sel=="3":
                return 
            else: print("쫌")

if __name__ == "__main__":
   bm = BankNumber()
   bm.main()