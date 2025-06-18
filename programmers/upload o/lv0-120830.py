# 코딩테스트 연습>코딩테스트 입문>짝수의 합
"""
정수 n이 주어질 때, n이하의 짝수를 모두 더한 값을 return 하도록 solution 함수를 작성해주세요.

제한사항
0 < n ≤ 1000

입출력 예
n	result
10	30
4	6
"""
def solution(n):
    #for나 while문 n되면 끝나게
    #더하기 공식은 외워라
    # i=0
    # s=0
    answer = 0
    i = 2
    while i<=n:
        answer+=i
        i+=2
    return answer
#     answer=0
#     for i in range(2, n+1, 2):
#         answer+= i
#     return answer
