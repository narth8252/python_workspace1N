import random

def game(user_rsp, computer_rsp, rsp):
    if computer_rsp == user_rsp:
        print(f'유져: {rsp[user_rsp]}, 컴퓨터: {rsp[computer_rsp]} => 비김')
        return "draw"
    elif ((computer_rsp == 1) and (user_rsp == 2)) or \
        ((computer_rsp == 2) and (user_rsp == 3)) or \
            ((computer_rsp == 3) and (user_rsp == 1)):
                
        print(f'유져: {rsp[user_rsp]}, 컴퓨터: {rsp[computer_rsp]} => user Win')
        return "user"
    elif ((computer_rsp == 2) and (user_rsp == 1)) or \
        ((computer_rsp == 3) and (user_rsp == 2)) or \
            ((computer_rsp == 1) and (user_rsp == 3)):
                
        print(f'유져: {rsp[user_rsp]}, 컴퓨터: {rsp[computer_rsp]} => computer Win')
        return "computer"       
        
    
def calc_prob(userWin, computerWin, rspCount):
    userWinprob = userWin / rspCount
    computerWinprob = computerWin /rspCount
        
    return userWinprob, computerWinprob

def output(userWin, computerWin, draw, rspCount):
    userWinprob, computerWinprob = calc_prob(userWin, computerWin, rspCount)
    print(f'유저 승: {userWin}, 컴퓨터 승: {computerWin}, 비김: {draw}, 유저 승률: {userWinprob*100}%, 컴퓨터 승률: {computerWinprob*100}%')

def main():
    rsp = {
        1: '가위',
        2: '바위',
        3: '보'
    }
    rspCount = 10
    userWin = 0
    computerWin = 0
    draw = 0

    for _ in range(rspCount):
        computer_rsp = random.randint(1, 3)
        while 1:
            user_rsp = input('1. 가위, 2. 바위, 3. 보 ')
            if user_rsp not in ['1', '2', '3']:
                print('잘못 입력됨')
            else:
                user_rsp = int(user_rsp)

                result = game(user_rsp, computer_rsp, rsp)
                if result == 'draw':
                    draw += 1
                elif result == 'user':
                    userWin += 1
                elif result == 'computer':
                    computerWin += 1
                break
        
    output(userWin, computerWin, draw, rspCount)
if __name__ == '__main__':
    main()