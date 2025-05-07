# 04-30 과제

# 가위바위보 게임
# 컴퓨터가 1 , 2 , 3 중에 랜덤값 하나를 생각하고 있음
# 1 = 가위 , 2 = 바위 , 3 = 보
# 컴퓨터 승 , 사람 승 , 무승부
# 10 번을 해서 각 승률을 계산 / 컴퓨터 몇 번 , 사람 몇 번 , 무승부 몇 번 했는지 나오도록

import random

# 전역변수 및 리스트
rpc = ["가위" , "바위" , "보"]
score = {"human_score" : 0 , "computer_score" : 0 , "draw" : 0}

# 유저가 입력하는 값 받는 함수
def inSert() :
    sertCheck_1 = False
    sertCheck_2 = False
    
    while sertCheck_1 == False or sertCheck_2 == False : 
        human = input("다음 값 중 하나를 입력하세요[가위(1) , 바위(2) , 보(3)] : ")
        sertCheck_1 = inSertCheck_1(human)
        if sertCheck_1 == True :
            sertCheck_2 = inSertCheck_2(human)
            if sertCheck_2 == True :
                return int(human)

# 유저가 엔터만 누르거나 2개 이상의 문자를 넣었을 때 오류 체크
def inSertCheck_1(number) :
    while len(number) > 1 or len(number) <= 0 :
        print("오입력되었습니다. 다시 입력해주세요.")
        return False
    return True

# 유저가 1 , 2 , 3 중에 하나만 입력했는 지 오류 체크
def inSertCheck_2(number) :
    while ord(number) < ord("1") or ord(number) > ord("3") :
        print("1 부터 3 까지의 숫자 하나만 입력하세요.")
        return False
    return True    

# 컴퓨터가 랜덤 정수를 생성하여 return 하는 함수
def inSertCom() :
    computer = random.randint(1,3)
    return computer

# 가위와 바위가 만났을 때
def rc(human , computer , s = score) :
    if human < computer :
        print(f"컴퓨터가 \"{rpc[computer - 1]}\" 를 냈고 , 유저가 \"{rpc[human - 1]}\" 를 내서 컴퓨터가 승리했습니다.")
        s["computer_score"] += 1
    else :
        print(f"유저가 \"{rpc[human - 1]}\" 를 냈고 , 컴퓨터가 \"{rpc[computer - 1]}\" 를 내서 유저가 승리했습니다.")
        s["human_score"] += 1
        
# 바위와 보가 만났을 때
def cp(human , computer , s = score) :
    if human < computer :
        print(f"컴퓨터가 \"{rpc[computer - 1]}\" 를 냈고 , 유저가 \"{rpc[human - 1]}\" 를 내서 컴퓨터가 승리했습니다.")
        s["computer_score"] += 1
    else :
        print(f"유저가 \"{rpc[human - 1]}\" 를 냈고 , 컴퓨터가 \"{rpc[computer - 1]}\" 를 내서 유저가 승리했습니다.")
        s["human_score"] += 1

# 보와 가위가 만났을 때
def pr(human , computer , s = score) :
    if human > computer :
        print(f"컴퓨터가 \"{rpc[computer - 1]}\" 를 냈고 , 유저가 \"{rpc[human - 1]}\" 를 내서 컴퓨터가 승리했습니다.")
        s["computer_score"] += 1
    else :
        print(f"유저가 \"{rpc[human - 1]}\" 를 냈고 , 컴퓨터가 \"{rpc[computer - 1]}\" 를 내서 유저가 승리했습니다.")
        s["human_score"] += 1

# 입력 받은 값으로 가위/바위/보를 판정할 수 있는 함수 생성
def verdict(human , computer , s = score) :
    if human == computer :
        s["draw"] += 1
        print(f"유저가 \"{rpc[human - 1]}\" 를 냈고 , 컴퓨터가 \"{rpc[computer - 1]}\" 를 내서 무승부가 되었습니다.")
    elif (human == 1 and computer == 2) or (human == 2 and computer == 1) :
        rc(human,computer)
    elif (human == 2 and computer == 3) or (human == 3 and computer == 2) :
        cp(human,computer)
    elif (human == 3 and computer == 1) or (human == 1 and computer == 3) :
        pr(human,computer)
        
# 메인 게임 시작
def main() :
    for i in range(0,10) :
        human_sel = inSert()
        computer_sel = inSertCom()
        verdict(human_sel,computer_sel)
    scoreOutput()
        
# 결과 출력
def scoreOutput() :
    print(f"컴퓨터 승리 : {score["computer_score"]} 회" , end="\t")
    print(f"유저 승리 : {score["human_score"]} 회" , end="\t")
    print(f"무승부 : {score["draw"]} 회")
    
    # 출력 후 결과 초기화
    for i in score :
        score[i] = 0

# 메인 메뉴
def menu() :
    sel = "0"
    while True :
        print("[1] 가위바위보 게임 시작")
        print("[0] 프로그램 종료")
        sel = input("메뉴를 선택하세요 : ")
        
        if sel   == "1" :
            main()
        elif sel == "0" :
            return
        else :
            print("메뉴 선택을 잘못 하셨습니다.")
            
menu()