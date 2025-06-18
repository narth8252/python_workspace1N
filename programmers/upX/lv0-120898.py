# 코딩테스트 연습>코딩테스트 입문>피자나눠먹기1
"""
머쓱이네 피자가게는 피자를 일곱 조각으로 잘라 줍니다. 
피자를 나눠먹을 사람의 수 n이 주어질 때, 
모든 사람이 피자를 한 조각 이상 먹기 위해 필요한 피자의 수를 return 하는 solution 함수를 완성해보세요.

제한사항
1 ≤ n ≤ 100

입출력 예
n	result
7	1
1	1
15	3

입출력 예 #1
7명이 최소 한 조각씩 먹기 위해서 최소 1판이 필요합니다.
입출력 예 #2
1명은 최소 한 조각을 먹기 위해 1판이 필요합니다.
입출력 예 #3
15명이 최소 한 조각씩 먹기 위해서 최소 3판이 필요합니다.
"""
import math

def solution(n):
    pizza = 1
    answer = pizza/7*n
    return math.ceil(answer)

"""
1. 파이썬에서 올림 함수 사용하기
파이썬에서는 math 모듈의 ceil 함수를 사용하면 됩니다.
import math
x = 3.2
result = math.ceil(x)  # 결과는 4

"""

#다른사람
def solution(n):
    return (n - 1) // 7 + 1

def solution(n):
    return -(-n//7)

def solution(n):
    return (n + 6) // 7

import math

def solution(n):
    return math.ceil(n/7)

from math import ceil
def solution(n):
    return ceil(n/7)
