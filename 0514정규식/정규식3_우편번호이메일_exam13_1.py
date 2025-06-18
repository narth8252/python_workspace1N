#0514 10시am exam13_1.py
#쌤ppt 8p

#우편번호 형식 맞추기 정규식예제
#6자리에서 5자리로 체계가 바뀜. 정수값입력받아서 우편번호 형식에 맞는지 확인하자
#우편번호 패턴방식: \d{5}$ #d정수{5}자리로 $끝나는 패턴만 가능하다.

"""
정수5자리만 입력가능합니다.
5자리 정수를 입력하면 "형식이 일치합니다."로 출력되고
그외의 문자나 자릿수가 다르면 "잘못된 형식입니다"가 출력됨.
"""
import re

zipcode = input("우편번호를 입력하세요")
pattern = r'\d{5}$'
regex = re.compile(pattern)
result = regex.match(zipcode)
if result != None:
    print("형식이 일치합니다.")
else:
    print("잘못된 형식입니다.")

#쌤ppt 9p.정규식함수
#점프투파이썬08-2 정규표현식시작하기

"""자주사용하는 패턴작성방법이다.
match(pattern, string)
    문자열의 시작부터 정규식과 매치되는지 검색조사.
    매칭되면 그정보를 저장한 객체반환
serch(pattern, string)
    문자열 전체를 검색해 정규식과 매치되는지 조사
    매칭되면 그 정보를 저장한 객체를 반환
findall(pattern, string)
    정규식과 매치되는 모든문자열을 iterator 객체(매칭객체)로 반환
    정규식과 매치되는 모든문자열(substring)을 list로 리턴
finditer(pattern, string)
    
    정규식과 매치되는 모든문자열(substr)을 iterator 객체로 리턴

    이터러블하다 하면 [리스트]다 배열구조다라고 이해하면됨
    이터러블이 배열구조는 아니지만 배열처럼 보인다.
    iterator(반복자) class내부에 실제 내부데이터와 아무상관없이 접근가능하다 
        클래스나 iterator 설계자가 만든다. 구축은 C에서 함.
        for문이 가능하다는 말임.
sub(pattern, replace, string, count=0, flag=0)
    정규식과 매치되는 모든문자열을 대체문자열로 교체하고 결과를 str타입으로 반환
 match, search는 정규식과 매치될때 match객체를 리턴, 매치X면 None리턴
 match객체란 정규식 검색결과로 리턴된 객체
"""

#쌤 ppt 10p에 colou?r  → "color", "colour" 에서 앞에 u없어야함.
# (x)colou?r  → "color", "colour" 에서 앞에 u없어야함.
# (o)colo?r  → "color", "colour" 
#12p. 컴파일해도되고 안해도 됨.
#match 함수는 첫부분에 star가 와야돼, 안오면 None
#match object

#쌤ppt-12p. match 함수 예제

import re

#match함수는 첫부분에  star가 와야한다. 이 문장에서 패턴못찾음.
text1 = " I like star" # "star"이 첫 부분에 오지 않으므로 패턴을 찾을 수 없습니다.
text2 = "star is beautiful" # "star"이 첫 부분에 오므로 패턴을 찾을 수 있습니다.

pattern = "star"

print (re.match( pattern, text1)) # None
print (re.match( pattern, text2)) # Match object

matchObj = re.match( pattern, text2)
print(matchObj.group()) # 찾은 패턴을 출력합니다.

#단어의 시작 및 종료 위치를 튜플로 나타냅니다.
#그룹함수를 통해 단어추출, 첫번째패턴의 시작위치 및 단어의종료위치, 단어위치값을 튜플로 나타냄
print(matchObj.start()) 
print(matchObj.end())
print(matchObj.span())

"""
    정규표현식
re 모듈을 사용하여 정규 표현식을 실행해서, 텍스트에서 패턴 찾기위해 사용
-match() 함수: 문자열의 첫 부분에서만 패턴을 찾습니다. 
  따라서 text1에서는 "star"이 첫 부분에 오지 않으므로 패턴을 찾을 수 없지만,
  text2에서는 "star"이 첫 부분에 오므로 패턴을 찾을 수 있습니다.
-match.group(): 찾은 패턴을 출력합니다. 이 경우 "star"을 출력합니다.
-match.start() 및 match.end(): 패턴의 시작 위치와 종료 위치를 각각 출력합니다. 예를 들어, "star"의 시작 위치는 0, 종료 위치는 4입니다.
-match.span(): 시작 위치와 종료 위치를 튜플로 반환합니다. 예를 들어, "star"의 경우 (0, 4)를 반환합니다.
이처럼 정규 표현식은 텍스트에서 특정한 패턴을 찾는 데 매우 유용합니다.

    이터러블과 리스트
이터러블은 반복 가능한 객체로, 리스트, 튜플, 딕셔너리 등이 있습니다. 리스트는 순서가 있는 변경 가능한 이터러블로, 배열 구조와 유사합니다. 하지만 Python에서는 배열보다는 리스트를 더 일반적으로 사용합니다.

결론
정규 표현식은 텍스트에서 특정 패턴을 찾는 데 사용되며, re.match() 함수는 문자열의 첫 부분에서만 패턴을 찾습니다. 이터러블은 리스트, 튜플 등이 있으며, 리스트는 배열 구조와 유사하지만 Python에서는 리스트를 더 많이 사용합니다.
"""

