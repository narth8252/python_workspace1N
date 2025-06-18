# 코딩테스트 연습> 코딩테스트 입문> n의 배수
"""
정수 n과 정수 배열 numlist가 매개변수로 주어질 때, 
numlist에서 n의 배수가 아닌 수들을 제거한 배열을 return하도록 solution 함수를 완성해주세요.

 제한사항
1 ≤ n ≤ 10,000
1 ≤ numlist의 크기 ≤ 100
1 ≤ numlist의 원소 ≤ 100,000

 입출력 예
n	numlist	                        result
3	[4, 5, 6, 7, 8, 9, 10, 11, 12]	[6, 9, 12]
5	[1, 9, 3, 10, 13, 5]	[10, 5]
12	[2, 100, 120, 600, 12, 12]	[120, 600, 12, 12
numlist에서 3의 배수만을 남긴 [6, 9, 12]를 return합니다.
"""
def solution(n, numlist):
    return [x for x in numlist if x % n == 0]
    #      numlist안에 x를 하나씩보며 n의배수인지?
    #                          x%n으로 나눠 나머지0이면,

def solution(n, numlist):
    result = []
    for num in numlist: #numlist안 숫자하나씩 차례대로꺼내 num변수에 넣어보자.
        if num%n == 0: #num이 n의 배수인지=num/n나머지가0이면
            result.append(num) #배수이면 result리스트에 append.
    return result

def solution(n, numlist):
    return list(filter(lambda x: x%n==0, numlist))
"""

"""


