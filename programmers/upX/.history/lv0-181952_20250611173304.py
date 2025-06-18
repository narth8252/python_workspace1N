# 코딩테스트 연습> 코딩 기초 트레이닝 > 문자열 출력하기

"""
문자열 str이 주어질 때, str을 출력하는 코드를 작성해 보세요.

입출력 예
a	b	flag	result
-4	7	true	3
-4	7	false	-11
"""
def solution(a, b, flag):
    if flag:
        return a+b
    else:
        return a-b
    
"""
boolean변수 : True 또는 False (대소문자 구분하며, 파이썬에서는 첫 글자 대문자)
"""

#다른풀이
