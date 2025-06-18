# 코딩테스트 연습>코딩테스트 입문>원소들의 곱과 합
"""
정수가 담긴 리스트 num_list가 주어질 때, 
모든 원소들의 곱이 모든 원소들의 합의 제곱보다 
작으면 1을 크면 0을 return

 입출력 예
num_list	    result
[3, 4, 5, 2, 1]	1
[5, 7, 8, 3]	0
모든 원소의 곱은 120, 합의 제곱은 225이므로 1을 return합니다.
"""
import math

def solution(num_list):
    total_sum = sum(num_list)
    total_prod = math.prod(num_list)

    if total_prod < total_sum**2:
        return 1
    else:
        return 0
"""
"""
#다른풀이
def solution(num_list):
    mul = 1
    for n in num_list:
        mul *= n
    return int(mul < sum(num_list) ** 2)

#다른풀이
import math

def solution(num_list):
    sq = sum(num_list) ** 2
    mul = math.prod(num_list)

    return 1 if mul < sq else 0

#틀림   
def solution(n):
    digit = list(map(int, str(n))) #숫자를 문자열로 바꾸고 각자리수를 int로 변환
    sq = sum(digit) ** 2 #자리합의 제곱

    mul = 1
    for d in digit:
        mul *= d #자리숫자곱셈

    if sq < mul:
        return 1
    else:
        return 0    
    