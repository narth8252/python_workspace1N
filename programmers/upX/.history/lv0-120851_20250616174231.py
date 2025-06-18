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
    for char in my_string:
        if char.isdigit():
            answer += char
    return answer

#정규표현식
import re
def solution(my_string):
    answer = re.findall(r'\d', my_string) 
    #\d는 하나이상의 숫자, re.findall()은 일치하는 숫자문자열을 전부 리스트로 반환함
    return answer

#문자열의 문자숫자개수 구하자
def solution(my_string):
    answer = 0
    for char in my_string:
        answer += 1
    return answer

#문자열의 숫자합
def solution(my_string):
    answer = 0
    for char in my_string:
        if char.isdigit(): #숫자면
            answer += int(char) #정수로 바꿔 더하기
    return answer

def solution(my_string):
    ret