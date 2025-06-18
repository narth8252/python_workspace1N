#0529 pm2:35
#이런짓 하지말고 있는거 써라
#객체지향의 장점: 연산자중복

class MyList:
    def __init__(self, data):
        self.data = list(data)

    def __str__(self):
        return f"MyList({self.data})"
    
    def __getitem__(self, index):
        #추상성
        if index>=0 and index < len(self.data):
            return self.data[index]
        return None
    
    def __setitem__(self, index, value):
        if index>=0 and index < len(self.data): 
            self.data[index]=value
    
m1 = MyList((1,2,3,4,5,6))
print(m1)
print(m1.data[0])
print(m1[0]) #객체內존재하는 배열을 외부에서 인덱스통해 접근해보자.
m1[0]=10
#나 index쓰거나 10없는거 쓰면 어쩔껀데->잡아줘야함(객체지향의 장점)