#쌤PPT-3p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# auto-mpg.csv
#파일명 : exam11_5.py
#데이터표준화
import re

pattern = r'비'
#re.compile을 해서 패턴을 내부객체에 등록
text = "하늘에 비가 오고 있습니다. 어제도 비가 왔고 오늘도 비가 오고 있습니다"
regex = re.compile(pattern) #패턴을 컴파일 시킨다 
#패턴에 일치하는 단어찾기
result = regex.findall(text)  #일치하는 단어를 모두 찾아서 리스트를 반환합니다 
print( result )

