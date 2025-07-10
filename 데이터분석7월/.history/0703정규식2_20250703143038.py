#쌤PPT-8p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_1.py
# ◈ 우편번호 형식 맞추기 정규식 예제
# 우리나라의 우편번호는 6자리에서 5자리로 체계가 바뀌었습니다. 정수 값을 입력을 받아서 이 데이터가 우편번호 형식에 맞는지 확인해보는 예제를 만들어 보겠습니다. 
# 우편번호 패턴 방식 : \d{5}$      <- 정수 5개만 가능하다 

import re

zipcode = input("우편번호를 입력하세요")               
pattern = r'\d{5}$'
regex = re.compile(pattern)
result = regex.match(zipcode)
if result != None:
    print("형식이 일치합니다.")
else:
    print("잘못된 형식입니다.")
