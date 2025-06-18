# 코딩테스트 연습> 코딩 기초 트레이닝 > n개 간격의 원소들

"""
정수 리스트 num_list와 정수 n이 주어질 때, num_list의 첫 번째 원소부터 마지막 원소까지 n개 간격으로 저장되어있는 원소들을 차례로 담은 리스트를 return하도록 solution 함수를 완성해주세요.

입출력 예
s	return
"a234"	false
"1234"	true

"""
def solution(s):
    # a = len(s) 변수선언하고 하면 더 빠르게 함
    # 문자열 길이가 4 또는 6이어야 함
    if len(s) == 4 or len(s) == 6:
        # 숫자로만 이루어졌는지 확인
        return s.isdigit()
    else:
        return False

"""
isdigit() 메서드란?
파이썬 문자열(String) 객체에서 사용할 수 있는 내장 메서드입니다.
문자열이 오직 숫자 문자(digit)로만 이루어져 있으면 True를 반환하고,

하나라도 숫자가 아닌 문자가 포함되어 있으면 False를 반환합니다.
특징 및 작동 원리
"1234".isdigit() → True (숫자로만 구성됨)
"12a4".isdigit() → False (문자 ‘a’가 포함됨)
" ".isdigit()→False` (공백은 숫자가 아님)
"-123".isdigit() → False (하이픈 ‘-’은 숫자가 아님)
즉, isdigit()은 부호나 소수점 등이 포함된 숫자 형태는 False로 처리합니다.
"""

#다른풀이
