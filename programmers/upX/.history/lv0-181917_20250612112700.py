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
""" C언어
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
max_value = (x > y) ? x: y;
            조건식    참 거짓
            절댓값이나 최솟값을 계산하는데 많이 쓰인다.
absolute_value = (x > 0) ? x: -x;	// 절대값 계산
max_value = (x > y) ? x: y;		// 최대값 계산
min_value = (x < y) ? x: y;		// 최소값 계산

x++, y++;  // x의 수식이 먼저 계산된 후 y의 수식이 계산됨
콤마 연산자는 한정된 공간에 여러 개의 문장을 넣을 때 사용된다. 
반복문이나 조건문에서 요긴하게 사용될 수 있다.

비트 단위 연산자
컴퓨터에서 모든 데이터는 비트로 표현된다. 비트(bit)는 컴퓨터에서 정보를 저장하는 가장 작은 단위이다.

비트는 0과 1 값만을 가진다.

비트 단위로 연산을 수행하고 다음과 같은 것들이 있다.

&	// 비트 AND	: 두 개의 피연산자의 해당 비트가 모두 1이면 1, 아니면 0
|	// 비트 OR	: 두 개의 피연산자의 해당 비트  중 하나라도 1이면 1, 아니면 0
^	// 비트 XOR	: 두 개의 피연산자의 해당 비트의 값이 같으면 0, 아니면 1
<<	// 왼쪽으로 이동	: 지정된 개수만큼 모든 비트를 왼쪽으로 이동한다.
>>	// 오른쪽으로 이동: 지정된 개수만큼 모든 비트를 오른쪽으로 이동한다.
~	// 비트 NOT	: 0은 1로 만들고 1은 0으로 만든다.
비트 단위 연산자는 정수 타입의 피연산자에만 적용할 수 있다.
참고) 정수 타입: char, short, int, long 등

비트 이동 연산자 <<, >>
x << y	// x의 비트들을 y칸만큼 왼쪽으로 비트 이동
x >> y	// x의 비트들을 y칸만큼 오른쪽으로 비트 이동
"""