#쌤PPT-18p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_6.py

import re

text = "I like star, red star, yellow star"
pattern = "star"
result = re.sub(pattern, "moon", text)
print(result) #문자열 전체 체인지 star->moon
# I like moon, red moon, yellow moon

result = re.sub(pattern, "moon", text, count=2)
print(result) #문자열 마지막 star만 moon으로 체인지
I like moon, red moon, yellow star

print("---------------------------------")
