# 코딩테스트 연습>코딩테스트 입문>배열의 유사도
"""
두 배열이 얼마나 유사한지 확인해보려고 합니다.
문자열 배열 s1과 s2가 주어질 때 같은 원소의 개수를
return하도록 solution 함수를 완성해주세요

 입출력 예
s1	                 s2	                        result
["a", "b", "c"]	["com", "b", "d", "p", "c"]	    2
["n", "omg"]	["m", "dot"]                	0
"b"와 "c"가 같으므로 2를 return합니다.
"""
def solution(s1, s2):
    for i in ()len(s1)

def solution(dot):
    # dot = x, y #좌측이 받아오는 변수값있어야함 튜플
    x, y = dot
    # x = dot[0]
    # y = dot[1]
    if x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:
        return 4

