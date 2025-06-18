# 코딩테스트 연습> 코딩테스트 입문> n의 배수 고르기
"""
정수 num과 n이 매개 변수로 주어질 때, num이 n의 배수이면 1을 return n의 배수가 아니라면 0을 return하도록 solution 함수
 제한사항
2 ≤ num ≤ 100
2 ≤ n ≤ 9

 입출력 예
num	n	result
98	2	1
34	3	0
98은 2의 배수이므로 1을 return합니다
"""
def solution(num, n):
    if num % n == 0:
        return 1
    else:
        return 0

"""

"""


