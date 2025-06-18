from random import randint
from typing import Literal

def rsp() -> Literal[-1, 0, 1]:
    computer = randint(1, 3)
    
    while True:
        user = input("1. 가위, 2. 바위, 3. 보\n> ")

        if user not in ["1", "2", "3"]:
            print("올바르지 않은 입력입니다.")
        else:
            break
    
    user = int(user)

    if computer == user:
        print("비겼습니다.")
        return 0
    elif computer % 3 == (user % 3) + 1:
        print("이겼습니다.")
        return 1
    else:
        print("졌습니다.")
        return -1

def main():
    # lose, draw, win
    stat = [0, 0, 0]
    games = 0

    while True:
        cmd = input("1. 가위바위보 하기\n2. 통계 보기\n3. 종료\n> ")

        if cmd == "1":
            res = rsp()
            stat[res+1] += 1
            games += 1
        elif cmd == "2":
            print(f"게임 수: {games}, 승률: {round(100 * stat[2] / games, 2)}%")
            print(f"승리: {stat[2]}, 무승부: {stat[1]}, 패배: {stat[0]}")
        elif cmd == "3":
            return
        else:
            print("올바르지 않은 입력입니다.")
            
main()