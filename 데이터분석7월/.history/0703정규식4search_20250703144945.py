#쌤PPT-11p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_1.py

# re.search()함수
import re

text1 = "I like star, red star, yellow star"
text2 = "starship is beautiful"

pattern = "star"

# 1. text1에서 "star"가 문자열 시작에 없지만, 첫 번째 "star" 위치에서 매치 객체를 반환
print (re.search( pattern, text1)) #None출력됨
# 2."starship is beautiful"에서 "star"가 문자열 시작에 있으므로 매치 객체 반환.
print (re.search( pattern, text2)) #<re.search object; span=(0, 4), search='star'>

# 3.매치 객체가 생성됨.
searchObj = re.search( pattern, text2)
# 4.매치된 문자열 "star" 반환.
print(searchObj.group() ) #star
# 5.매치 시작 위치(0) 반환.
print(searchObj.start() ) #0
# 6.매치 끝 위치(4) 반환.
print(searchObj.end() ) #4
# 7.매치된 구간의 튜플 (0, 4) 반환.
print(searchObj.span() ) #(0, 4)
# 8.text2의 [:4]는 인덱스0부터3까지(총4글자)
print(text2[:4]) #star

# re.match()는 문자열의 시작에서만 패턴을 찾음
# re.search()는 문자열 전체에서 패턴을 찾아 처음 발견되는 위치에서 매치 객체를 반환함


