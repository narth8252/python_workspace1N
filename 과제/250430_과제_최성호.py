import random

recordList = []
record = {}

def init():
    record["gameCnt"] = 0
    record["humanWin"] = 0
    record["comWin"] = 0
    record["draw"] = 0
    record["humanLose"] = 0
    record["comLose"] = 0
    recordList.append(record)
    
def comValue():
    global comHand
    comHand = random.randint(1, 3)
    return comHand
    
def humanValue():
    global humanHand
    humanHand = int(input("패를 입력하세요: (가위: 1 / 바위: 2 / 보자기: 3)"))
    if (humanHand >= 1 and humanHand <= 3) == False:
        print("잘못된 숫자를 입력하셨습니다. 1부터 3까지의 숫자만 입력해주세요.")
    return humanHand
    
def decision(humanHand):
    if humanHand == 1:
        humanScissors(humanHand)
    elif humanHand == 2:
        humanRock(humanHand)
    else:
        humanPaper(humanHand)
    
def humanScissors(humanHand):
    if humanHand == 1:
        record["gameCnt"] += 1
        if comHand == 1:
            print("사용자: 가위 / 컴퓨터: 가위  ==> 무승부!!")
            record["draw"] += 1
        elif comHand == 2:
            print("사용자: 가위 / 컴퓨터: 바위  ==> 컴퓨터 승리!!")
            record["comWin"] += 1
            record["humanLose"] += 1
        else:
            print("사용자: 가위 / 컴퓨터: 보자기  ==> 컴퓨터 승리!!")
            record["comWin"] += 1
            record["humanLose"] += 1

def humanRock(humanHand):
    if humanHand == 2:
        record["gameCnt"] += 1
        if comHand == 1:
            print("사용자: 바위 / 컴퓨터: 가위  ==> 사용자 승리!!")
            record["comLose"] += 1
            record["humanWin"] += 1
        elif comHand == 2:
            print("사용자: 바위 / 컴퓨터: 바위  ==> 무승부!!")
            record["draw"] += 1
        else:
            print("사용자: 바위 / 컴퓨터: 보자기  ==> 컴퓨터 승리!!")
            record["comWin"] += 1
            record["humanLose"] += 1

def humanPaper(humanHand):
    if humanHand == 3:
        record["gameCnt"] += 1
        if comHand == 1:
            print("사용자: 보자기 / 컴퓨터: 가위  ==> 컴퓨터 승리!!")
            record["comWin"] += 1
            record["humanLose"] += 1
        elif comHand == 2:
            print("사용자: 보자기 / 컴퓨터: 바위  ==> 사용자 승리!!")
            record["comLose"] += 1
            record["humanWin"] += 1
        else:
            print("사용자: 보자기 / 컴퓨터: 보자기  ==> 무승부!!")
            record["draw"] += 1

def endGame():
    record["humanWinRate"] = record["gameCnt"] / record["humanWin"]
    record["comWinRate"] = record["gameCnt"] / record["comWin"]
    print(f"총 게임 수: {record["gameCnt"]}판 / 사용자 승: {record["humanWin"]}판 / 사용자 패: {record["humanLose"]}판 / 무승부: {record["draw"]}판 / 승률: {(record["humanWin"] / record["gameCnt"] * 100):.2f}%")

def playGame():
    comValue()
    humanValue()
    decision(humanHand)

def rcpGame():
    init()
    while True:
        sel = int(input(f"게임을 진행하시려면 1을, 종료하시려면 0을 입력하세요: "))
        if sel == 1:
            playGame()
        elif sel == 0:
            print()
            endGame()
            print("게임을 종료합니다. 이용해주셔서 감사합니다.")
            return
        else:
            print("잘못된 숫자를 입력하셨습니다. 0과 1중 하나를 선택해 다시 입력해주세요.")

if __name__ == "__main__":
    rcpGame()