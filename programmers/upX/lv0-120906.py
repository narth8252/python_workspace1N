# 코딩테스트 연습>코딩테스트 입문>분수의 덧셈


"""
정수 n이 매개변수로 주어질 때 n의 각 자리 숫자의 합을 
return하도록 solution 함수를 완성해주세요

 제한사항
0 ≤ n ≤ 1,000,000

 입출력 예
n	  result
1234	10
930211	16
1 + 2 + 3 + 4 = 10을 return합니다.
"""
def solution(n):
    return sum(int(digit) for digit in str(n))
#str(n)을 이용해 각 자리 숫자를 하나씩 꺼내고, int로 바꿔주시면 됩니다.
# str(n)으로 숫자를 문자열로 바꾸고
# 그 문자열을 한 글자씩 꺼내 for digit in str(n)
# 각 글자를 int로 변환해서 int(digit)
# digit은 "숫자 한 자리"
# 모두 더합니다. sum()
"""
정수를 문자열로 변환하면 각 자리 숫자를 문자 하나씩 다룰 수 있습니다.
문자열의 각 문자를 다시 정수로 변환하여 더하면 됩니다.
"""

#다른풀이
def solution(n):
    return sum(int(i) for i in str(n))

def solution(n):
    n = str(n)
    answer = 0
    for i in n:
        answer +=int(i)
    return answer

def solution(n):
    return sum(list(map(int, str(n))))
# str(n)으로 정수를 문자열로 변환합니다.
# map(int,str(n))는 문자열의 각 문자를 int 함수에 적용해 숫자 리스트로 만듭니다.
# list()로 그 결과를 리스트로 만들고 (list는 생략 가능)
# sum()으로 모든 숫자를 더합니다.
# list()는 sum()에 꼭 필요하지 않으므로, 더 간단히 작성할 수도 있습니다.
def solution(n):
    return sum(map(int, str(n)))

"""
map함수: 파이썬 내장 함수 중 하나로, 특정 함수를 여러 데이터에 한꺼번에 적용할 때 사용합니다.
첫 번째 인자로 함수, 두 번째 인자로 반복 가능한 자료형(리스트, 문자열 등)을 받습니다.
입력 자료형의 각 요소에 함수를 적용한 결과를 차례대로 반환합니다.
"""
numbers = ['1','2','3']
result = map(int, n)
print(list(result)) #[1, 2, 3]


