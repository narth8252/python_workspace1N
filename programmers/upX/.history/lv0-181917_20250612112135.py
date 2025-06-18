# 코딩테스트 연습> 코딩 기초 트레이닝 > 간단한논리연산
"""
boolean 변수 x1, x2, x3, x4가 매개변수로 주어질 때, 
다음의 식의 true/false를 return 하는 solution 함수를 작성해 주세요.
(x1 ∨ x2) ∧ (x3 ∨ x4)

입출력 예
x1	    x2  	x3  	x4	    result
false	true	true	true	true
true	false	false	false	false
예제 1번의 x1, x2, x3, x4로 식을 계산하면 다음과 같습니다.
(x1 ∨ x2) ∧ (x3 ∨ x4) ≡ (F ∨ T) ∧ (T ∨ T) ≡ T ∧ T ≡ T
따라서 true

"""
def solution(x1, x2, x3, x4):
    if x1 == False and x2 == False:
        x1x2 = False
    else:
        x1x2 = True
    if x3 == False and x4 == False:
        x3x4 = False
    else:
        x3x4 = True
    if x1x2 == True and x3x4 == True:
        return True
    else:
        return False
    
#다른풀이
