import random

game_results = {"computer": 0, "person": 0, "draw": 0}

def append():
    for i in range(10):
        try:
            person = int(input("가위 바위 보(1~3): "))
            print(f"You: {['가위','바위','보'][person - 1]}")
            if person not in [1, 2, 3]:
                print("1~3만 입력해주세요.")
                continue
        except:
            print("숫자를 입력해주세요.")
            continue
        
        computer = random.randint(1, 3)
        print(f"Computer: {['가위','바위','보'][computer - 1]}")
        
        if person == computer:
            print("무승부")
            game_results["draw"] += 1
        elif (person == 1 and computer == 3) or \
             (person == 2 and computer == 1) or \
             (person == 3 and computer == 2):
             print("사람")
             game_results["person"] += 1
        else:
            print("컴퓨터")
            game_results["computer"] += 1

def results():
    print("최종결과")
    print(f"사람 승: {game_results['person']}번")
    print(f"컴퓨터 승: {game_results['computer']}번")
    print(f"무승부: {game_results['draw']}번")

def main():
    append()
    results()

main()