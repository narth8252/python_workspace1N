#ppt 13p
#serch 함수

#쌤ppt-12p. search 함수 예제

import re

#search함수는 첫부분에  star가 와야한다. 이 문장에서 패턴못찾음.
text1 = " I like star" # "star"이 첫 부분에 오지 않으므로 패턴을 찾을 수 없습니다.
text2 = "star is beautiful" # "star"이 첫 부분에 오므로 패턴을 찾을 수 있습니다.

pattern = "star"

print (re.search( pattern, text1)) # None
print (re.search( pattern, text2)) # search object

searchObj1 = re.search(pattern, text1)
searchObj2 = re.search(pattern, text2)
# 찾은 패턴을 출력합니다.
print(searchObj1.group()) #star
print(searchObj2.group()) #star

#단어의 시작 및 종료 위치를 튜플로 나타냅니다.
#그룹함수를 통해 단어추출, 첫번째패턴의 시작위치 및 단어의종료위치, 단어위치값을 튜플로 나타냄
print(searchObj1.start()) #7
print(searchObj2.start()) #0

print(searchObj1.end())   #11
print(searchObj2.end())   #4

print(searchObj1.span())  # (7, 11)
print(searchObj2.span())  # (0, 4)

#위 예시에서 text1와 text2 모두에서 "star" 패턴을 찾을 수 있습니다.
# 이 예시에서 text1에서는 "star"이 문자열의 중간에 위치해 있으며, 시작 위치는 7이고 종료 위치는 11입니다. text2에서는 "star"이 문자열의 처음에 위치해 있으며, 시작 위치는 0이고 종료 위치는 4입니다.

# 이처럼 re.search() 함수는 문자열의 어디에서든지 패턴을 찾을 수 있으므로, 첫 부분에만 패턴이 있어야 하는 것은 아닙니다.

#   결론
# re.search() 함수는 문자열의 어디든지 패턴을 찾을 수 있습니다. 따라서 문장의 첫 부분에만 패턴이 있어야 하는 것은 아닙니다. re.match() 함수는 문자열의 첫 부분에서만 패턴을 찾습니다.
#   re.search() 와 re.match()의 차이
# re.match(): 문자열의 첫 부분에서만 패턴을 찾습니다. 따라서 "star"이 첫 부분에 오지 않으면 패턴을 찾을 수 없습니다.
# re.search(): 문자열의 어디든지 패턴을 찾을 수 있습니다. 따라서 "star"이 문자열의 어디에 위치해 있든지 패턴을 찾을 수 있습니다.
