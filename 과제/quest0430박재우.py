import random

score  = []
cases = {1 : [0, -1, 1], 2 : [1, 0, -1], 3 : [-1, 1, 0]}

def display():
    return input('0 종료, 다른 값 계속 : ')

def show_rock_paper_shears():
    return input('1(가위), 2(바위), 3(보) 중 하나를 숫자로 골라주세요 : ')

def display_result(choice, r):
    print(f'{"1(가위)" if choice == 1 else "2(바위)" if choice == 2 else "3(보)"}를 낸 결과 ' + '이겼습니다' if r > 0 else '비겼습니다' if r == 0 else '졌습니다')

def main():
    while True:
        if display() == '0': break
        user_choice = int(show_rock_paper_shears())
        result = cases[user_choice][random.randint(0, 2)]
        display_result(user_choice, result)
        score.append(result)
    print(f'{len(score)}번의 승부 중 {score.count(1)}번 승리한 승률 {sum(score) / len(score)}의 좋은 게임이었습니다.')

main()
        