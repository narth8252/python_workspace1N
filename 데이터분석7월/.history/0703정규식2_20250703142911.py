#쌤PPT-8p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# auto-mpg.csv
#파일명 : exam11_5.py
#데이터표준화
import re
import re
 
zipcode = input("우편번호를 입력하세요")               
pattern = r＇\d{5}$’ 
regex = re.compile(pattern)
result = regex.match(zipcode)
if result != None:
    print("형식이 일치합니다.")
else:
    print("잘못된 형식입니다.")
