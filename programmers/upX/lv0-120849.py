# 코딩테스트 연습>코딩테스트 입문>배열 자르기
"""
정수 배열 numbers와 정수 num1, num2가 매개변수로 주어질 때, 
numbers의 num1번 째 인덱스부터 num2번째 인덱스까지 자른 정수 배열을 
return 하도록 solution 함수를 완성해보세요.

입출력 예
numbers	        num1	num2	result
[1, 2, 3, 4, 5]	1	    3	   [2, 3, 4]
[1, 3, 5]   	1	    2	   [3, 5]
[1, 2, 3, 4, 5]의 1번째 인덱스 2부터 3번째 인덱스 4 까지 자른 [2, 3, 4]를 return 합니다.
"""
def solution(numbers, num1, num2):
    answer = numbers[num1 : num2+1]
    return answer

"""

"""

#다른풀이
