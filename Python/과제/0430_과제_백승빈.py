import random
win = 0
draw = 0
lose = 0
uc = 0
cc = 0
choices = {1: "가위" ,2:"바위",3:"보" }

def mmm():
    return print(f"당신 ({choices[uc]}) vs 컴퓨터 ({choices[cc]})")

def userchoice():
    return int(input()) 

def comchoice():
    return random.randint(1,3)

def game():
    global win, draw, lose, uc ,cc      
    while True:
        print("1(가위) 2(바위) 3(보) 0(게임종료)중 하나를 입력해주세요")
        uc = userchoice()
        cc = comchoice()           
        if uc == 0:  
          print("게임을 종료합니다.")
          return 
       
        if (uc == 1 and cc == 3) or (uc == 2 and cc == 1) or (uc == 3 and cc == 2):
            mmm()
            print("승리하였습니다")
            win += 1
        elif cc == uc:
            mmm()
            print("무승부입니다!")
            draw += 1
        elif cc >= uc or (cc == 1 and uc == 3):
            mmm()
            print("패배하였습니다!")
            lose += 1 
        else : print("잘못 선택하셨습니다")
        
        print(f"당신의 전적:  {win} 승 {lose} 패 {draw} 무")
       
game()