# 코딩테스트 연습>코딩테스트 입문>배열의 평균값

"""
정수 배열 numbers가 매개변수로 주어집니다. 
numbers의 원소의 평균값을 return하도록 solution 함수를 완성해주세요.

제한사항
0 ≤ numbers의 원소 ≤ 1,000
1 ≤ numbers의 길이 ≤ 100
정답의 소수 부분이 .0 또는 .5인 경우만 입력으로 주어집니다.

입출력 예
numbers	                                        result
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]	                5.5
[89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]    94.0
"""
def solution(numbers):
    answer = sum(numbers)/len(numbers)
    return answer

# import numpy as np
# def solution(numbers):
#     return np.mean(numbers)

# def solution(numbers):
#     return sum(numbers) / len(numbers)


# def solution(numbers):
#     answer = 0
#     count = len(numbers)
#     for i in range(count):
#         answer += numbers[i]
#     return answer/count

# def solution(numbers):
#     answer = 0
#     for i in numbers:
#         answer += i
#     return answer / len(numbers)


# def solution(numbers):
#     answer = 0
#     return sum(numbers)/len(numbers)

# def solution(numbers):
#     answer = 0
#     for i in numbers:
#         answer += i
#     answer /= len(numbers)
#     return answer
