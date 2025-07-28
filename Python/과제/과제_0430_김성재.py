import random

def start():
    print("가위, 바위, 보를 시작합니다.")

def userChoice():
        choice = int(input("1.가위, 2.바위, 3.보 중 하나를 입력하세요: "))
        if choice in [1, 2, 3]:
            return choice
        else:
            print("1~3 중에서 선택하세요.")
            return userChoice()
        
def winner(user, computer):
    if user == computer:
        return 1  # 무승부
    elif (user == 1 and computer == 3) or (user == 2 and computer == 1) or (user == 3 and computer == 2):
        return 3  # 사람 승
    else:
        return 2  # 컴퓨터 승

def playgame():
    start()
    userWinCount = 0
    comWinCount = 0
    drawCount = 0
    put = {1: "가위", 2: "바위", 3: "보"}

    for i in range(1, 11):
        print(f"\n{i}번째 게임")
        user = userChoice()
        computer = random.randrange(1, 4)
        result = winner(user, computer)

        print(f"당신: {put[user]}, 컴퓨터: {put[computer]}")

        if result == 1:
            print("결과: 무승부")
            drawCount += 1
        elif result == 2:
            print("결과: 컴퓨터 승")
            comWinCount += 1
        else:
            print("결과: 사람 승")
            userWinCount += 1

    print("\n게임 종료")
    print(f"사람 승: {userWinCount}, 컴퓨터 승: {comWinCount}, 무승부: {drawCount}")
    print(f"사람 승률: {userWinCount / 10:.2f}")
    print(f"컴퓨터 승률: {comWinCount / 10:.2f}")

playgame()