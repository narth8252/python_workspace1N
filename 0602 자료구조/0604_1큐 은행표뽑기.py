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