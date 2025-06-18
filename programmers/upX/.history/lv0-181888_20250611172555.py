# 코딩테스트 연습> 코딩 기초 트레이닝 > n개 간격의 원소들

"""
두 정수 a, b와 boolean 변수 flag가 매개변수로 주어질 때, flag가 true면 a + b를 false면 a - b를 return 하는 solution 함수를 작성해 주세요.

입출력 예
a	b	flag	result
-4	7	true	3
-4	7	false	-11

"""
def solution(num_list, n):
    answer = num_list[::n]
    return answer

"""

"""

#다른풀이
