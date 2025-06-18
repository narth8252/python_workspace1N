# 코딩테스트 연습> 코딩테스트 입문> 이어 붙인 수
"""
정수가 담긴 리스트 num_list가 주어집니다. 
num_list의 홀수만 순서대로 이어 붙인 수와 
짝수만 순서대로 이어 붙인 수의 합을 return

 제한사항
2 ≤ num_list의 길이 ≤ 10
1 ≤ num_list의 원소 ≤ 9
num_list에는 적어도 한 개씩의 짝수와 홀수가 있습니다.

 입출력 예
num_list	    result
[3, 4, 5, 2, 1]	393         351+42
[5, 7, 8, 3]	581
홀수만 이어 붙인 수는 351이고 짝수만 이어 붙인 수는 42입니다. 
두 수의 합은 393입니다.
"""

def solution(num_list):
    result1 = []
    result2 = []
    
    for n in num_list:
        if n%2 == 0: #n이짝수면
            result2.append(str(n)) #str(n)문자열로 변환
        else:
            result1.append(str(n))

    n1 = int(''.join(result1)) if result1 else 0  # 홀수로 만든 수
    n2 = int(''.join(result2)) if result2 else 0  # 짝수로 만든 수
    
    return n1 + n2  # 두 수의 합 반환

#다른풀이
def solution(num_list):
    answer = 0
    even = "" #홀수
    odd = "" #짝수
    for n in num_list:
        if n%2!=0: #홀수면
            even+=str(n) #홀수변수even안에 문자열로 넣어라
        else:
            odd+=str(n)
    return int(even) + int(odd) #정수로바꿔 더해라
"""
"""
# 짝수홀수개수
def solution(num_list):
    answer = [0,0] #인덱스1에 짝수개수, 인덱스2에 홀수개수 선언
    for n in num_list:
        answer[n%2]+=1 #나머지가 0일땐 인덱스1, 나머지가1일땐 인덱스2에 1씩 증가
    return answer

#짝수의합
def solution(n):
    answer = 0
    i = 2
    while i<=n:
        answer+=i
        i+=2
    return answer
