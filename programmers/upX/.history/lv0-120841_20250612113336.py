# 코딩테스트 연습>코딩테스트 입문>점의 위치 구하기
def solution(dot):
    # (x,y)
    if x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    if x < 0 and y < 0:
        return 3
    if x > 0 and y < 0:
        return 4
    # answer = 0
