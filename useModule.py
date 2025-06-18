#모듈사용 0513.10:48am
#main.py 파일을 생성하고, 다음 코드를 입력하여 모듈을 사용합니다.

import module

print("--------구분:useModule.py 프린트---------------")
print(module.add(3, 4))  
print(module.sub(3, 4))

print("--------구분:useModule.py 프린트---------------")
p2 = module.Person("윤하", 44)
p2.print()

"""출력:
--------구분:module.py 프린트---------------
7
-1
--------구분:module.py 프린트---------------
name=조승연 age=30
--------구분:useModule.py 프린트---------------
7
-1
--------구분:useModule.py 프린트---------------
name=윤하 age=44

"""
"""
import module as md #모듈명 길 경우 aliasing 다른이름으로 부르기가능

print(md.add(3, 4))  
print(md.sub(3, 4))

print("--------파일명을 md로 줄임:useModule.py 프린트---------------")
p2 = md.Person("웬디", 32)
p2.print()

#자주사용
print("----파일명쓰기싫고 내 함수처럼 쓰고싶을때---")
from module import add, sub 
print( add(9,8))
print( sub(9,8))

from module import md
P3 = Person("조이", 24)
p3.print()
"""

#수학라이브러리 -> 머신러닝은 numpy타입으로만 전환
#파이썬은?
#파일명은 프로그램(파이썬)이 제공하는 함수나 클래스명으로 만들면 절대안됨.
#numpy.py나 pandas.py 등.. 자기껄 자꾸 들고와서 오류남
#파일명을 바꾸면 됨
print("--------파이썬문법(스칼라연산1:1)--------------")
import numpy as np
a = [1,2,3,4,5,6,7,8,9,10]
b = [x*2 for x in a ]
c = a+b
print(a)
print(b)
print(c)

#numpy사용?
print("--------numpy사용-------------")
a1 = np.array(a)
b1 = 2 * a1
c1 = a1 +b1
print(a1)
print(b1)
print(c1)

# 1. 벡터 연산:다대 다 연산가능(원래는 1:1연산)
# 벡터 연산은 각 벡터의 요소끼리 대응되는 성분끼리 연산하는 것을 말합니다. 
# NumPy 배열은 벡터 연산을 매우 간단하고 빠르게 처리할 수 있도록 지원합니다
#NumPy 배열끼리는 요소별로 연산이 자동으로 수행됩니다
print("--------벡터연산(다:다)-------------")
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
# 벡터 덧셈
add_result = v1 + v2  # 결과: [5 7 9]
# 벡터 뺄셈
sub_result = v1 - v2  # 결과: [-3 -3 -3]
# 벡터 곱셈 (요소별 곱)
mul_result = v1 * v2  # 결과: [4 10 18]
# 벡터 나눗셈 (요소별 나누기)
div_result = v1 / v2  # 결과: [0.25 0.4  0.5]

# 2. 스칼라 연산(1:1연산)
# 스칼라 연산은 벡터의 각 요소에 하나의 숫자(스칼라)를 곱하거나 더하는 연산입니다.
#벡터의 모든 요소에 스칼라 연산이 간단히 적용됩니다.
print("-------스칼라연산(1:1)-------------")
import numpy as np
v = np.array([1, 2, 3])
# 스칼라 곱셈
scalar_mul = v * 3   # [3 6 9]
# 스칼라 덧셈
scalar_add = v + 5   # [6 7 8]

#3. 벡터 내적 (스칼라 결과)
#벡터의 내적은 두 벡터 요소별 곱의 합이며, 결과는 스칼라입니다.
#np.dot 함수로 쉽게 계산할 수 있습니다
print("-------벡터내적(스칼라결과)-------------")
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 결과: 1*4 + 2*5 + 3*6 = 32

"""
스칼라연산 - 1:1 연산, 대부분 프로그래밍언어가 기본사용
            다:다 연산은 for문을 써야했다.
통계분석용 언어들은 수학에 가깝게 벡터연산 지원
벡터연산 - 다:다 연산. R, python의 numpy, pandas 라이브러리가 지원

연산 종류	NumPy표현	      설명
벡터 덧셈	v1 + v2  	     요소별 덧셈
벡터 뺄셈	v1 - v2	         요소별 뺄셈
요소 곱셈	v1 * v2	         요소별 곱셈
스칼라곱셈	v * scalar       벡터의 모든 요소에 스칼라 곱
내   적	   np.dot(v1, v2)	벡터내적, 결과는 스칼라
이처럼 NumPy는 반복문없이도 벡터와 스칼라연산 처리. 
"""

#문제. mymodule2.py 파일명만들어서.
#내부구조 이해하라고.. 모듈에서 불러오기 예제
#isEven(4) 짝수면True, 홀수면 False 반환
#toUpper('asterisk') 대문자로 반환 -> ASTERISK

import mymodule2



#1. isEven() 함수
#짝수 True, 홀수 False반환. 
#간단하게 모듈로 % 연산자 사용하여 짝수여부 확인.
#num을 2로 나눈 나머지가 0이면 True, 아니면 False
def isEven(num):
    return num %2 ==0

#2. toUpper() 함수
# 파이썬의내장함수 upper() 문자열을 대문자로 변환.
def toUpper(s):
    return s.upper()

# 짝수 확인
print(isEven(4))  # True
print(isEven(3))  # False

# 대문자 변환
print(toUpper('asterisk'))  # ASTERISK