# 코딩테스트 연습>숨어있는 숫자의 덧셈 (1)
"""
문자열 my_string이 매개변수로 주어집니다. 
my_string안의 모든 자연수들의 합

 입출력 예
my_string	    result
"aAb1B2cC34oOp"	10
"1a2b3c4d123"	16
"""
#문자하나씩 돌면서 숫자만 수집
def solution(my_string):
    answer = ''
    for char in my_string():
        if char.isdigit():
            resulf += char
    return answer

#정규표현식
import re
def solution(my_string):
    numbers = re.findall(r)

