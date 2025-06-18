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
"""
https://cwithvisual.tistory.com/entry/%EB%85%BC%EB%A6%AC-%EC%97%B0%EC%82%B0%EC%9E%90-%EC%A1%B0%EA%B1%B4-%EC%97%B0%EC%82%B0%EC%9E%90-%EC%BD%A4%EB%A7%88-%EC%97%B0%EC%82%B0%EC%9E%90-%EB%B9%84%ED%8A%B8-%EB%8B%A8%EC%9C%84-%EC%97%B0%EC%82%B0%EC%9E%90-%EC%97%B0%EC%82%B0%EC%9E%90%EC%9D%98-%EC%9A%B0%EC%84%A0-%EC%88%9C%EC%9C%84%EC%99%80-%EA%B2%B0%ED%95%A9-%EA%B7%9C%EC%B9%99
논리 연산자
x && y 	// AND 연산
x || y 	// OR 연산
!x 	// NOT 연산
- x && y : AND 연산 : x와 y가 모두 참일 경우에 참
- x || y : OR 연산 : x가 참이거나 y가 참일 경우에 참 <-> x와 y가 모두 거짓일 경우에만 거짓
- !x : NOT 연산 : x가 참이면 거짓 <-> 거짓이면 참

조건 연산자 : 유일하게 3개의 피연산자를 가지는 삼항 연산자이다.
최댓값을 계산하는 수식을 적어보면 다음과 같다.

max_vlue = (x > y) ? x: y;
"""