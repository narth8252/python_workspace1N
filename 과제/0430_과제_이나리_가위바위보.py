#쌤풀이0502 가위바위보 게임:통계
import random #import 외부파일(모듈)을 이 파일로 갖고와라
#거듭해서 import 해도 한번만 들어옴
#게임개시: gameStart()

# 출력값 :컴터승1, 사람승2, 무승부3 판단함수
# 입력값(전역변수보다 매개변수추천) : 매개변수보는 컴퓨터의 생각, 사람의 입력값

#무승부부터 판단하는게 빠름
def isWinner(computer, person):
    if computer == person: #사람=컴퓨터
        return 3 #무승부
    
    #컴퓨터가 이기는 경우의 수 or
    #상수지정: 가위1 바위2 보3
    #\역슬래시는 여러줄에 문장 작성시 같은줄이다 라는 의미로 \ 앞뒤로 공백필요
    if (computer==1 and person==3) or \
       (computer==2 and person==1) or \
       (computer==3 and person==2):
        return 1 #컴퓨터가 이김
    
    #사람 이기는 경우의 수 or
    return 2 #어차피3경우뿐이라 굳이 if, else로 따질필요X

#코드간결하게 하고 번호1,2,3 대신 가위바위보로 나오게 tiles 추가해준다.
titles = ["", "가위", "바위", "보"]
titles2 = ["", "컴퓨터승", "사람승", "무승부"]
gameList=[] #{"computer":"값", "person":"값", winner:"값"}

def test():
    for i in range(0, 10):
        computer = random.randint(1,3)
        person = random.randint(1,3)
        winner = isWinner(computer, person)
        print("컴퓨터:", titles[computer], "사람:",titles[person], titles2[winner])

def gameStart():
    gameList.clear() #.clear"함수호출:데이터만 삭제
    #계속 반복
    while True:
        computer = random.randint(1,3)
        person = int(input("1.가위 2.바위 3.보 "))
        winner = isWinner(computer, person)
        print("컴퓨터:", titles[computer],"사람:", titles[person], titles2[winner])
        gameList.append({"computer":computer, "person":person,
                         "Winner:":winner})
        again = input("게임을 계속하시겠습니까? y/n ")
        if again !="Y" and again!="y":
            return

def gameStatistic():

    cpmputer_win=0
    person_win=0
    equak_win=0
    for game in gameList:
        if game["winner"]=="1":
            computer_win+=1
        elif game["winner"]=="2":
            person_win+=1
        else:
            equal_win+=1 #비김draw


    for game in gameList:
        print(f"컴퓨터: {game["computer"]}", end="\t")
        print(f"사람: {game["person"]}", end="\t")
        print(f"승패: {game["winner"]}")
    print("컴퓨터 승", computer_win)
    print("사람 승", person_win)
    print("무승부", equal_win)


#gameStart() #함수 호출
def gameMain():
    while True:
        print("1.게임시작")
        print("2.게임통계")
        print("3.게임종료")
        sel = input("선택 : ")
        if sel=="1":
            gameStart() 
        elif sel=="2":
            gameStatistic()
        elif sel=="3":
            print("게임을 종료합니다.")
            return

gameMain()

"""
import random

def start():
    print("가위,바위,보")

def userChoice():
    choice = int(input("1.가위, 2.바위, 3.보: "))
    if choice in [1, 2, 3]:
        return choice
    else:
        print("1~3 중에 선택하세요")
        return userChoice()
    
def winner(user, computer):
    if user == computer:
        return 1 #무승부
    elif (user==1 and computer==3) or (user==2 and computer==1) or (user==3 and computer==2):
        return 3 #사람승
    else:
        return 2#컴터승
    
def playGame():
    start()
    userWinCount =0
    comWinCount = 0
    drawCount = 0
    put = {1: "가위", 2: "바위", 3:"보"}

    for i in range(1, 11):
        print(f"\n{i}번째게임")
        user = userChoice()        
        computer = random.randrange(1, 4)
        result = winner(user, computer)

        print(f"당신: {put[user]}, 컴퓨터: {put[computer]}")

        if result==1:
            print("결과: 무승부")
            drawCount +=1
        elif result ==2:
            print("결과: 컴퓨터 승")
            comWinCount +=1
        else:
            print("결과: 사람 승")
            userWinCount +=1

    print("\게임종료")
    print(f"사람 승: {userWinCount}, 컴퓨터 승: {comWinCount}, 무승부: {drawCount}")
    print(f"사람 승률: {userWinCount/ 10:.2f}")
    print(f"컴퓨터 승률: {comWinCount/ 10:.2f}")

playGame()
"""
'''
# 가위바위보 게임
# 컴퓨터가 1,2,3중에 랜덤값 하나를 생각하고 있음
# 사람이 1.가위 2.바위 3.보 입력
# 컴퓨터승 사람승 무승부 출력
# 게임 10번 해서 승률 출력까지
# 컴퓨터 3  사람 2  무승부 5
import random

RockScissorPaper = ["가위", "바위", "보"]
playerDict = {"Win": 0, "Draw": 0, "Lose": 0}

#메뉴보이기
def menuDisplay():
    print("1.게임")
    print("2.승률")
    print("0.종료")

#승패 확인
def isWin(computer, player):
    print(f"플레이어: {RockScissorPaper[player-1]}, 컴퓨터: {RockScissorPaper[computer-1]}")
    if player == computer:
        playerDict["Draw"] += 1
        print("무승부")
    elif (player == 1 and computer == 2) or (player == 2 and computer == 3) or (player == 3 and computer == 1):
        playerDict["Lose"] += 1
        print("컴퓨터 승")
    elif (player == 1 and computer == 3) or (player == 2 and computer == 1) or (player == 3 and computer == 2):
        playerDict["Win"] += 1
        print("플레이어 승")
    print()

#플레이
def gamePlay():
    while True:
        select = input("1.가위, 2.바위, 3.보 >>")
        if select != "1" and select != "2" and select != "3":
            print("잘못 입력하셨습니다.")
        else:
            break

    select = int(select)
    computer = random.randint(1,3)
    isWin(computer, select)

def winDisplay():
    print(f"컴퓨터: {playerDict["Lose"]}, 플레이어: {playerDict["Win"]}, 무승부: {playerDict["Draw"]}")
    print()

def start():
    while True:
        menuDisplay()
        select = input(">>")
        if select =="1":
            #게임하기
            gamePlay()
        elif select =="2":
        #출력하기
            winDisplay()
        elif select =="0":
            print("게임 종료")
            break
        else:
            print("잘못 입력하셨습니다.")

    print("end")

start()
'''
# from random import randint
# from typing import Literal

# def rsp() -> Literal[-1, 0, 1]:
#     computer = randint(1, 3)

#     while True:
#         user = input("1. 가위, 2. 바위, 3. 보\n> ")

#         if user not in ["1", "2", "3"]:
#             print("올바르지 않은 입력입니다.")
#         else:
#             break

#     user = int(user)

#     if computer == user:
#         print("비겼습니다.")
#         return 0
#     elif computer % 3 == (user % 3) + 1:
#         print("이겼습니다.")
#         return 1
#     else:
#         print("졌습니다.")
#         return -1
    
# def main():
#     #lose, draw, win
#     stat = [0, 0, 0]
#     games = 0

#     while True:
#         cmd = input("1.게임\n 2.승률\n 3.종료\n> ")      

#         if cmd == "1":
#             res = rsp()
#             stat[res+1] += 1
#             games += 1
#         elif cmd == "2":
#             print(f"게임 수: {games}, 승률: {round(100 * stat[2]/games, 2)}%")
#             print(f"승리: {stat[2]}, 무승부: {stat[1]}, 패배: {stat[0]}")
#         elif cmd == "3":
#             return
#         else:
#             print("올바르지 않은 입력입니다.")

# main()