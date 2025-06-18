import random
#가위바위보 게임
#컴퓨터가 1,2,3중에 랜덤값 하나를 생각하고 있음.
#사람이 1.가위, 2.바위, 3.보 중에 선택. 입력받아서 컴퓨터승 사람승 무승부
#10번해서 승률 컴퓨터 3 사람 2 무승부 5

def ComSelect():
    comSelect = random.randrange(1,3)
    print("컴퓨터 결정 완료")
    
    return comSelect

def PlayerSelect():
    while(True):
        playerSelect = int(input("숫자 입력(1~3) : "))
        
        if(playerSelect == 1 or playerSelect == 2 or playerSelect == 3):
            print("입력 완료")
            return playerSelect
        else:
            print("1~3의 숫자를 입력하세요.")

#1.가위, 2.바위, 3.보
def Duel(comSelect,playerSelect):
    if (comSelect == 1 and playerSelect == 1) or (comSelect == 2 and playerSelect == 2) or (comSelect == 3 and playerSelect == 3):
        return 1 #무승부
    elif (comSelect == 1 and playerSelect == 2) or (comSelect == 2 and playerSelect == 3) or (comSelect == 3 and playerSelect == 1):
        return 2 #플레이어 승
    elif (comSelect == 1 and playerSelect == 3) or (comSelect == 3 and playerSelect == 2) or (comSelect == 2 and playerSelect == 1):
        return 3 #컴퓨터 승


def WinCount(resultDuel, winCount):
    if resultDuel == 1:
        winCount["draw"] += 1
    elif resultDuel == 2:
        winCount["player"] += 1
    elif resultDuel == 3:
        winCount["com"] += 1

    return winCount

def WinResult(winCount,gameNum):

    comWinAvg = winCount["com"] / gameNum
    playerWinAvg = winCount["player"] / gameNum
    drawAvg = winCount["draw"] / gameNum

    WinResultPrint(winCount,comWinAvg,playerWinAvg,drawAvg)

    return

def WinResultPrint(winCount, comWinAvg, playerWinAvg, drawAvg):
    print(f"컴퓨터 승리 횟수 : {winCount["com"]}, 평균 : {comWinAvg:.2f}")
    print(f"플레이어 승리 횟수 : {winCount["player"]}, 평균 : {playerWinAvg:.2f}")
    print(f"비긴 횟수 : {winCount["draw"]}, 평균 : {drawAvg:.2f}")
    
    return 0

def Game():
    gameNum = 10
    winCount= {"com":0,"player":0,"draw":0}

    print("게임을 시작합니다.")
    for i in range(0,gameNum):
        comAction = ComSelect()
        playerAction = PlayerSelect()
        print(f"{comAction},{playerAction}")
        resultDuel = Duel(comAction, playerAction)
        winCount = WinCount(resultDuel,winCount)

        print("....")

    print("결과를 정산합니다.")
    WinResult(winCount, gameNum)
    
    print("타이틀로 돌아갑니다.")

    return

def StartPrint():
    print("메뉴 선택")
    print("1. 게임 시작")
    print("0. 게임 종료")

    return 0

def Switch():
    while(True):
        sel = input("메뉴 입력 : ")
        if sel == '1':
            print("게임 시작")
            Game()
        elif sel == '0':
            print("프로그램 종료")
            exit()
        else:
            print("잘못된 입력")
            StartPrint()
        #타이틀 돌아온 후 다시 출력
        StartPrint()

    return 0

def main():

    StartPrint()
    Switch()

    return 0

main()