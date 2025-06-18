#0430mun_kimsunwoo_research
import random 


connectkey = {1:"가위", 2:"바위", 3:"보"}

winloseequal = [0, 0, 0]

for i in range(10):
    print("1.가위 2.바위 3.보")
    sel = int(input("당신의 선택 =) "))
    compu = random.randint(1,3)
    print("컴퓨터의 수 ) ", connectkey.get(compu))
    print("당신의 수 ) ",  connectkey.get(sel))
    if sel == compu:
        print("무승부")
        winloseequal[0] += 1
    elif (sel) == (compu%3+1):
        print("player Win")
        winloseequal[1] += 1
    else:
        print("player Lose")
        winloseequal[2] += 1
    
print("무승부 횟수", winloseequal[0])
print("승리 횟수", winloseequal[1])
print("패배 횟수", winloseequal[2])