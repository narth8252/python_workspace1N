# 코딩테스트 연습>코딩테스트 입문>중앙값 구하기

"""
정수 배열 numbers가 매개변수로 주어집니다. 
numbers의 각 원소에 두배한 원소를 가진 배열을 
return하도록 solution 함수를 완성해주세요.

 제한사항
-10,000 ≤ numbers의 원소 ≤ 10,000
1 ≤ numbers의 길이 ≤ 1,000

 입출력 예
numbers                 	result
[1, 2, 3, 4, 5]	            [2, 4, 6, 8, 10]
[1, 2, 100, -99, 1, 2, 3]	[2, 4, 200, -198, 2, 4, 6]
[1, 2, 3, 4, 5]의 각 원소에 두배를 한 배열 [2, 4, 6, 8, 10]을 return합니다.
"""
def solution(numbers):
    return [num*2 for num in numbers]
            #새 리스트로[각원소를2배, for in 배열 순회]반환

def solution(numbers):
    return [2*int(digit) for digit in str(numbers)]
print(solution(123))  # [2, 4, 6]
print(solution(405))  # [8, 0, 10]
#파이썬에선 작동. 프로그래머스에서 오류
#ValueError: invalid literal for int() with base 10: '['
#int('[')처럼 숫자가 아닌 문자열을 정수로 바꾸려 했다는 뜻
#int() 함수가 [ 같은 문자를 보고 "이건 숫자로 바꿀 수 없어!" 하고 에러
#입력을 문자열 형태로 잘못 받았거나, str(n)이 리스트 형태의 문자열일 때 발생
"""


"""

#다른풀이
def solution(numbers):
    answer = []

    for i in numbers:
        answer.append(i*2)
    return answer

def solution(numbers):
    answer = []
    for i in range(len(numbers)):
        answer.append(numbers[i]*2)
    return answer

