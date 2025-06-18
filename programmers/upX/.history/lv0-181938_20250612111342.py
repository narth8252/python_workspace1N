# 코딩테스트 연습> 코딩 기초 트레이닝 > 두 수의 연산값 비교하기

"""
boolean 변수 x1, x2, x3, x4가 매개변수로 주어질 때, 다음의 식의 true/false를 return 하는 solution 함수를 작성해 주세요.

(x1 ∨ x2) ∧ (x3 ∨ x4)

입출력 예
x1	x2	x3	x4	result
false	true	true	true	true
true	false	false	false	false
예제 1번의 x1, x2, x3, x4로 식을 계산하면 다음과 같습니다.
(x1 ∨ x2) ∧ (x3 ∨ x4) ≡ (F ∨ T) ∧ (T ∨ T) ≡ T ∧ T ≡ T
따라서 true

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
