# 코딩테스트 연습>코딩테스트 입문>점의 위치 구하기
def solution(dot):
    dot = x, y
    if x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:
        return 4
    # answer = 0
