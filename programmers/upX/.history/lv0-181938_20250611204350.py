# 코딩테스트 연습> 코딩 기초 트레이닝 > 두 수의 연산값 비교하기

"""
연산 ⊕는 두 정수에 대한 연산으로 두 정수를 붙여서 쓴 값을 반환합니다. 
예를 들면 다음과 같습니다.

12 ⊕ 3 = 123
3 ⊕ 12 = 312
양의 정수 a와 b가 주어졌을 때, a ⊕ b와 2 * a * b 중 더 큰 값을 
return하는 solution 함수를 완성해 주세요.
단, a ⊕ b와 2 * a * b가 같으면 a ⊕ b를 return 합니다.

a	b	result
2	91	364
91	2	912

"""
# str_input = input().strip()
# n = int(input().strip())

str_input, n = input().strip().split()
n = int(n)
result = str_input * n
print(result)
"""
input() 사용자로부터 한 줄의 문자열을 입력받는 함수
.strip() 입력받은 문자열의 양쪽 끝에 있는 공백과 줄바꿈 문자 등을 제거합니다.
.split() 문자열을 공백(띄어쓰기, 탭 등)을 기준으로 나누어, 나누어진 부분들을 리스트로 만듭니다.
str_input, n = ...
나누어진 리스트의 요소들을 각각 변수 str_input과 n에 차례대로 할당합니다.
즉, 리스트에 정확히 두 개의 요소가 있어야 합니다.
"""

#다른풀이
