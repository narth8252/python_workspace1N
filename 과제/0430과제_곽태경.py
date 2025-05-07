import random
rsp={1:"가위",2:"바위",3:"보"}
def game():
    myNum=int(input("1가위 2바위 3보 입력하세요 :"))
    key=random.randint(1,3)
    if key-myNum==-1 or key-myNum==2:
        print(f"사람이 이겼습니다!컴퓨터: {rsp[key]} 사람:{rsp[myNum]}")
        return 0
    elif key==myNum:
        print(f"무승부입니다!컴퓨터: {rsp[key]} 사람:{rsp[myNum]}")
        return 1
    else :
        print(f"컴퓨터가 이겼습니다!컴퓨터: {rsp[key]} 사람:{rsp[myNum]}")
        return 2
win=[0,0,0]
for i in range(10):
    win[game()]+=1
print(f"컴퓨터{win[2]}승 사람{win[0]}승 무승부{win[1]}번")
print(f"너의 승률은 {win[0]*10}% 이야")