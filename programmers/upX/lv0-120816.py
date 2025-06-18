# 코딩테스트 연습>코딩테스트 입문>배열 원소의 길이


"""
머쓱이네 피자가게는 피자를 두 조각에서 열 조각까지 원하는 조각 수로 잘라줍니다. 
피자 조각 수 slice와 피자를 먹는 사람의 수 n이 매개변수로 주어질 때, 
n명의 사람이 최소 한 조각 이상 피자를 먹으려면 
최소 몇 판의 피자를 시켜야 하는지를 return 하도록 solution 함수를 완성해보세요.

입출력 예
slice	n	result
7	10	2
4	12	3
10명이 7조각으로 자른 피자를 한 조각 이상씩 먹으려면 최소 2판을 시켜야 합니다.
"""
#풀이 10명이 7조각 = 2판
#n//slice ->정수올림 math.ceil(num)
#n//slice+1 -> 소수점자르기 int(num)
import math

def solution(slice, n):
    answer = math.ceil(n / slice)  
    # 일반 나누기를 사용하여 실수 결과 도출 후 올림 처리
    return answer

"""

"""

#다른풀이
def solution(slice, n):
    return((n-1)//slice) + 1

from math import ceil
def solution(slice, n):
    return ceil(n/slice)