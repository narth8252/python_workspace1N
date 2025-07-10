#0707 pm1시 윤년
#입력데이터[[1],[2],[3]...]
# 출력데이터 [0,0,0,1,0,0,0,1....]
#1~20205

def isLeap(year):
    if year%4==0 and year%100!= 0 or year%400==0:
        return True
    return False

X = []  
y = []