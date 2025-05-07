# 가위바위보 게임
# 컴퓨터가 1, 2, 3 중 하나를 랜덤으로 선택한다. 
# 사용자도 1, 2, 3 중 하나를 선택한다. 
# 컴퓨터가 사용자보다 1, 2, 3 중 하나를 랜덤으로 선택한다. 
# 결과를 출력한다. 
# 1:가위, 2:바위, 3:보 
import random

def game_play():
    print("\n 컴퓨터와 가위바위보 게임을 시작합니다.")
    com = random.randint(1,3)
    user = int(input("1:가위, 2:바위, 3:보 : "))
    
    # 컴퓨터가 가위, 바위, 보를 랜덤으로 선택한다. str로 변환한다.
    com_str = {1:"가위", 2:"바위", 3:"보"}[com]
    # 사용자가 가위, 바위, 보를 랜덤으로 선택한다.
    user_str = {1:"가위", 2:"바위", 3:"보"}[user]
    # 결과를 출력한다.
    print(f"컴퓨터는 {com}를 냈습니다.")
    print(f"사용자는 {user}를 냈습니다.")

    # 승패를 출력한다.
    if com == user:
        print("비겼습니다.")
        return "draw"
    elif (com == 1 and user == 3) or (com == 2 and user == 1) or (com == 3 and user == 2):
        print("사용자가 이겼습니다.")
        return "user"
    else:
        print("컴퓨터가 이겼습니다.")
        return "com"
    
# 가위바위보 게임을 10번해서 승률을 계산한다. 컴퓨터 : 3, 사람 : 2, 무승부 : 5
def game_rating():
    user_win = 0
    com_win = 0
    draw = 0

    for i in range(10):
        print(f"\n[{i+1}번재 게임]")
        result = game_play()
        if result == "user":
            user_win += 1
        elif result == "com":
            com_win += 1
        else:
            draw += 1
    print("\n======================게임 결과 요약======================")
    print(f"컴퓨터 : {com_win}점, 사람 : {user_win}점, 무승부 : {draw}점")
    print(f"컴퓨터의 승률은 {(com_win / 10) * 100}% 입니다.")

game_rating()
