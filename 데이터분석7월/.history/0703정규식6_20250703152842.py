#쌤PPT-14p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_1.py
"""
"""

import re

text1 = "I like star, red star, yellow star"
pattern = "star"
result = re.sub(pattern, "moon", text)
print(result) #문자열 전체 체인지 star->moon

result = re.sub(pattern, "moon", text, count)
print(result) #문자열 전체 체인지 star->moon


# 각 줄마다 이름, 전화번호, 이메일 추출

print("---------------------------------")
