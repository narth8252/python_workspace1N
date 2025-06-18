# 코딩테스트 연습>코딩테스트 입문>점의 위치 구하기
"""
x 좌표와 y 좌표가 모두 양수이면 제1사분면에 속합니다.
x 좌표가 음수, y 좌표가 양수이면 제2사분면에 속합니다.
x 좌표와 y 좌표가 모두 음수이면 제3사분면에 속합니다.
x 좌표가 양수, y 좌표가 음수이면 제4사분면에 속합니다.
입출력 예
dot	    result
[2, 4]	1
[-7, 9]	2

"""
def solution(dot):
    # dot = x, y #좌측이 받아오는 변수값있어야함 튜플
    x, y = dot
    x = dot[0]
    y = dot[1]
    if x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:
        return 4

