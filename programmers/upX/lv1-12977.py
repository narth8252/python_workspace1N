# 코딩테스트 연습> 코딩테스트 입문> 소수만들기
"""
주어진 숫자 중 3개의 수를 더했을 때 소수가 되는 경우의 개수를 구하려고 합니다. 
숫자들이 들어있는 배열 nums가 매개변수로 주어질 때, 
nums에 있는 숫자들 중 서로 다른 3개를 골라 더했을 때 소수가 되는 경우의 개수를 
return 하도록 solution 함수를 완성해주세요.

 제한사항
nums에 들어있는 숫자의 개수는 3개 이상 50개 이하입니다.
nums의 각 원소는 1 이상 1,000 이하의 자연수이며, 중복된 숫자가 들어있지 않습니다.

 입출력 예
nums	    result
[1,2,3,4]	1
[1,2,7,6,4]	4
[1,2,4]를 이용해서 7을 만들 수 있습니다.
"""

#쌤0611
#1.소수인지 물어봄, 소수란 1과 자기자신으로만 나눌수있음.
def isPrime(num): 
    i=2
    # if num==1 or num ==2:
    #     return False
    while i<=num/2: #1 2 3 4 ...100 절반까지만 나눠보면 뒤는 맞을것임.
        if num%i ==0: #/2해서 나머지0이면 소수아니므로 return하고 함수종료
            return False
        i += 1
    return True #마지막까지 남으면 소수임
#루트num에 약수는 항상짝으로 있으니까

#2.3중 for루프 돌림: num리스트에서 3개 숫자 i,j,k골라 합이 소수인지 판단
#중복비허용 돌릴때 1,2,7,6,4     1 2 7 6 4      1 2 4 6 7
def solution(nums):
    answer = 0
    for i in range(0, len(nums)-2):
    #i가 0부터 len(nums)-3까지 반복 →리스트에서 첫번째숫자 선택
        for j in range(i+1, len(nums)-1): #중복허용안하니 i+1,허용하면
        #j가 i+1부터 len(nums)-2까지 반복 →첫번째 다음위치~세번째숫자 선택
            for k in range(j+1, len(nums)):
            #k가 j+1부터 len(nums)-1까지 반복 →두번째 다음위치~세번째숫자 선택
                if isPrime(nums[i]+nums[j]+nums[k]): #3숫자의합이 소수라면
                   #print(nums[i],nums[j],nums[k])
                   answer+=1   #조합의 개수를 1개씩 더해라
    return answer 
"""
함수 목적
nums 리스트에서 3개 숫자를 골라서 그 합이 소수인지 판단합니다.
합이 소수인 조합의 개수를 answer에 더합니다.
3중 for문의 역할
첫 번째 for문: i가 0부터 len(nums)-3까지 반복

→ 리스트에서 첫 번째 숫자 선택
두 번째 for문: j가 i+1부터 len(nums)-2까지 반복

→ 첫 번째 숫자 다음 위치부터 두 번째 숫자 선택
세 번째 for문: k가 j+1부터 len(nums)-1까지 반복

→ 두 번째 숫자 다음 위치부터 세 번째 숫자 선택
이렇게 해서 중복 없이 서로 다른 인덱스의 3개 숫자 조합만 선택합니다.

예시: nums = [1, 2, 7, 6, 4]일 때
len(nums) = 5 이므로,
i는 0부터 2까지 (0, 1, 2)
j는 i+1부터 3까지
k는 j+1부터 4까지 반복합니다.
조합이 어떻게 생성되는지 살펴보겠습니다.

i	j	k	선택 숫자들	합
0	1	2	1, 2, 7	10
0	1	3	1, 2, 6	9
0	1	4	1, 2, 4	7
0	2	3	1, 7, 6	14
0	2	4	1, 7, 4	12
0	3	4	1, 6, 4	11
1	2	3	2, 7, 6	15
1	2	4	2, 7, 4	13
1	3	4	2, 6, 4	12
2	3	4	7, 6, 4	17
소수 여부 판단
7, 11, 13, 17은 소수입니다.
따라서 합이 소수인 조합은 (1,2,4), (1,6,4), (2,7,4), (7,6,4) 네 가지입니다.
결과적으로 answer = 4가 됩니다.

요약
3중 for문은 리스트에서 인덱스가 겹치지 않는 3개 숫자 조합 모두를 탐색합니다.
각 조합의 합을 isPrime 함수로 검사하여 소수면 카운트합니다.
"""
#print( isPrime(12))
print( solution([1,2,3,4]) )

# https://wikidocs.net/21638 에라토스테네스의체

#풀이2-1.3개 숫자조합구하기
# from itertools import combinations

# nums = [1, 2, 3, 4, 5]
# for comb in combinations(nums, 3):
#     print(comb)

#풀이2-2.소수판별
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

#풀이2-3.조합 합이 소수인지 확인후 개수세기
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
