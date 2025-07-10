#쌤PPT-11p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_1.py

# re.match()함수
import re

text1 = "I like star"
text2 = "starship is beautiful"

pattern = "star"

# 1. "I like star"에서 "star"가 문자열 시작에 없으므로 None 반환.
print (re.match( pattern, text1)) #None출력됨
# 2."starship is beautiful"에서 "star"가 문자열 시작에 있으므로 매치 객체 반환.
print (re.match( pattern, text2)) #<re.Match object; span=(0, 4), match='star'>

# 3.매치 객체가 생성됨.
matchObj = re.match( pattern, text2)
# 
print(matchObj.group() ) #star
print(matchObj.start() ) #0
print(matchObj.end() ) #4
print(matchObj.span() ) #(0, 4)

print(text2[:4])


