# 코딩테스트 연습>코딩테스트 입문>분수의 덧셈
"""
첫 번째 분수의 분자와 분모를 뜻하는 numer1, denom1, 
두 번째 분수의 분자와 분모를 뜻하는 numer2, denom2가 매개변수로 주어집니다.
두 분수를 더한 값을 기약 분수로 나타냈을 때 분자와 분모를 
순서대로 담은 배열을 return 하도록 solution 함수를 완성해보세요.

 제한사항
0 <numer1, denom1, numer2, denom2 < 1,000

 입출력 예
numer1	denom1	numer2	denom2	result
1	    2   	3	    4	    [5, 4]
9	    2	    1	    3   	[29, 6]
1 / 2 + 3 / 4 = 5 / 4입니다. 따라서 [5, 4]를 return 합니다.
"""
import math
def solution(numer1, denom1, numer2, denom2):
    numerator = numer1*denom2+numer2*denom1
    denominator = denom1*denom2
    gcd = math.gcd(numerator, denominator)
    answer = [numerator//gcd, denominator//gcd]
    return answer
"""
math.gcd 함수는 파이썬 표준 라이브러리인 math 모듈에 포함되어 있으며, 두 수의 최대공약수를 쉽게 구할 수 있습니다.
// 연산자는 정수 나눗셈(몫)입니다.
gcd = math.gcd(numerator, denominator) #최대공약수 구하기
"""

#다른풀이
import math
def solution(denum1, num1, denum2, num2):
    denum = denum1 * num2 + denum2 * num1
    num = num1 * num2
    gcd = math.gcd(denum, num)
    return [denum//gcd, num//gcd]

#분모가 다른 분수의 계산 함수
from fractions import Fraction

def solution(denum1, num1, denum2, num2):
    f = Fraction(denum1, num1) + Fraction(denum2, num2)
    answer = [f.numerator, f.denominator]
    return answer


