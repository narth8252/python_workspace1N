# 코딩테스트 연습> 코딩테스트 입문> 개미 군단

"""
소수점 아래 숫자가 계속되지 않고 유한개인 소수를 유한소수라고 합니다. 
분수를 소수로 고칠때 유한소수로 나타낼수 있는 분수인지 판별하려고 합니다. 
유한소수가 되기 위한 분수의 조건은 다음과 같습니다.

기약분수로 나타내었을 때, 분모의 소인수가 2와 5만 존재해야 합니다.
두 정수 a와 b가 매개변수로 주어질 때, 
a/b가 유한소수이면 1을, 무한소수라면 2를 return하도록 
solution 함수를 완성해주세요.

 입출력 예
a	b	result
7	20	1
11	22	1
12	21	2
분수7/20은 기약분수 입니다. 분모 20의 소인수가 2, 5 이기 때문에 유한소수입니다. 따라서 1을 return합니다.
"""
















#1.3개 숫자조합구하기
# from itertools import combinations

# nums = [1, 2, 3, 4, 5]
# for comb in combinations(nums, 3):
#     print(comb)

#2.소수판별
#소수:1과 자기자신만 나눠떨어지는 2이상의 양의정수
#7을 나누었을 때 나누어 떨어지는 수는 1과 7뿐. 2~6으로 나누면 안떨어짐.
#계산: 2부터 n의 제곱근까지만 검사하여 나눠떨어지는 수가 있는지?
# import math
# def is_prime(n):
#     if n < 2:
#         return False #2이상의 수만 소수
#     for i in range(2, int(math.sqrt(n)) + 1):
#         if n%i == 0:
#             return False #나눠떨어지는 수는 소수아님
#     return True #나누어떨어지는 수없으면 소수

#3.조합 합이 소수인지 확인후 개수세기
#조합을 하나씩 꺼내 합구한뒤, 소수인지 판단
#소수라면 결과 개수를 1씩 증가

#최종 : 소수 판별 효율을 위해 제곱근까지만 검사
nums = [1,2,7,6,4]
from itertools import combinations #as cb:이후 코드에서 combinations() 대신 cb()를 사용하실 수 있습니다.
import math

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num))+1):
                  #(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def solution(nums):
    cnt = 0
    for comb in combinations(nums, 3):
        total = sum(comb)
        if is_prime(total):
            cnt += 1
    return cnt

#다른풀이1.
def solution(nums):
    from itertools import combinations as cb
    answer = 0
    for i in cb(nums, 3):
        prime_sum = sum(i)
        for j in range(2, prime_sum):
            if prime_sum%j==0:
                break
        else:
            answer += 1
    return answer


#  itertools 모듈은 반복(iteration) 작업을 도와주는 여러 가지 효율적인 함수를 포함
# combinations는 리스트나 튜플같은 자료형에서 순서에 무관하게 특정개수만큼 뽑는 모둔 경우의 수를 만들어주는 함수
#[1,2,3]에서 2개씩 뽑는 조합을 모두 구하면 (1,2),(1,3),(2,3)
