# 코딩테스트 연습>코딩테스트 입문>공배수

"""
정수 number와 n, m이 주어집니다. number가 n의 배수이면서 m의 배수이면 1을 아니라면 0을 return하도록 solution 함수를 완성해주세요.
"""
def solution(number, n, m):
    if number % n == 0 and number % m == 0: #나머지가0이면 배수임.
        return 1
    else:
        return 0

