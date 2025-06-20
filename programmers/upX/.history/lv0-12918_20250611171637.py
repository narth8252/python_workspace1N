# 코딩테스트 연습> 연습문제 >문자열 다루기 기본

"""
문자열 s의 길이가 4 혹은 6이고, 숫자로만 구성돼있는지 확인해주는 함수, 
solution을 완성하세요. 
예를 들어 s가 "a234"이면 False를 리턴하고 "1234"라면 True를 리턴하면 됩니다.

입출력 예
s	return
"a234"	false
"1234"	true

"""
def solution(s):
    # 문자열 길이가 4 또는 6이어야 함
    if len(s) == 4 or len(s) == 6:
        # 숫자로만 이루어졌는지 확인
        return s.isdigit()
    else:
        return False

"""

"""

#다른풀이
