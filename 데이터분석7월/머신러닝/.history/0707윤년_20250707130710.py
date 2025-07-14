#0707 pm1시 윤년
#입력데이터[[1],[2],[3]...]
# 출력데이터 [0,0,0,1,0,0,0,1....]
#1~20205

def isLeap(year):
    if year%4==0 and year%100!=0 or year%400==0:
        return 1
    return 0

X = []  
y = []
#데이터 생성 조작할때는 리스트가 편함
for i in range(1, 2026):
    X.append(i)
    y.append(isLeap(i))