# 코딩테스트 연습>코딩테스트 입문>배열 자르기
"""
문자열 s의 길이가 4 혹은 6이고, 숫자로만 구성돼있는지 확인해주는 함수, 
solution을 완성하세요. 
예를 들어 s가 "a234"이면 False를 리턴하고 "1234"라면 True를 리턴하면 됩니다.

입출력 예
s	return
"a234"	false
"1234"	true
문제가 잘 안풀린다면😢
"""
def solution(numbers, num1, num2):
    answer = numbers[num1 : num2+1]
    return answer

"""

"""

#다른풀이
