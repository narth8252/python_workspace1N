# 코딩테스트 연습 > 코딩테스트 입문 > 짝수 홀수 개수

"""
어떤 자연수를 제곱했을 때 나오는 정수를 제곱수라고 합니다. 
정수 n이 매개변수로 주어질 때, n이 제곱수라면 1을 
아니라면 2를 return하도록 solution 함수를 완성해주세요.

제한사항
1 ≤ n ≤ 1,000,000

입출력 예
n	result
144     1
976	    2
"""
def solution(n):
    i = 1
    while i **2 <= n:          # i의 제곱이 n보다 크기 전까지 반복합니다
        if i **2 == n:         # i의 제곱이 n와 같으면
            return 1           # 제곱수입니다
        i += 1                 # i값을 1씩 증가시켜서 계속 검사
    return 2                     # 제곱수가 아니면 1이나 다른 값 반환
# **제곱수(제곱), **0.5제곱근(루트)=sqrt

#     if n**0.5 == int(n**0.5) :
#         return 1
#     else :
#         return 2
    

#     if n**2 == int(n**2) :
#         return 1
#     else :
#         return 2
#     이 코드는 n을 제곱하는 것(n**2)이 정수인지를 검사하는 것처럼 보여집니다. 그러나,

# n**2은 어디까지나 n이 어떤 값이든 제곱을 하는 연산이고,
# 이 결과는 항상 실수(심지어 정수일지라도 기계의 계산 방식에 따라 float 타입일 수 있음)입니다.
# 즉, 이 조건은 항상 참이거나 기대와는 다른 결과를 낼 수 있습니다.

# 왜 그럴까요?
# n이 정수라면, n**2도 정수이고, int(n**2)도 동일한 값이기 때문에,
# 이론적으로는 항상 참이 되어야 하는데,
# 하지만, 문제가 생길 수 있습니다:
# n이 부동소수점(실수) 타입일 경우, n**2 역시 부동소수점이 되며,
# int(n**2)는 소수점 이하를 자르고 정수로 바꿔서 비교하는데,
# 부동소수점 연산의 특성상 근사값이 발생하여 원래의 정확한 값과 차이가 생길 수 있습니다.
# import math

# def solution(n):
#     root = math.sqrt(n)
#     if root.is_integer():
#         return 2  # 제곱근인 경우
#     else:
#         return 1  # 제곱수가 아닌 경우

# 다른풀이
def solution(n):
    return 1 if (n ** 0.5).is_integer() else 2

def solution(n):
    return 1 if (n ** 0.5) % 1 == 0 else 2

import math
def solution(n):
    return 1 if int(math.sqrt(n)) ** 2 == n else 2