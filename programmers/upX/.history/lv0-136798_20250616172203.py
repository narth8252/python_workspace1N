# 코딩테스트 연습>기사단원의 무기
"""
정수 n이 매개변수로 주어질 때, n 이하의 홀수가 오름차순으로 담긴 배열

 입출력 예
n	result
10	[1, 3, 5, 7, 9]
15	[1, 3, 5, 7, 9, 11, 13, 15]
"""
def solution(n):
    return [i for i in range(1, n+1, 2)]

def solution(n):
    answer = []
    if n%2 != 0:
        answer.append()
    return answer

print(solution(10))

def solution(n):
    result = []
    i = 1
    while i <= n:
        result.append(i)
        i += 2
    return result