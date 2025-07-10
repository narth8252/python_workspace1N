#쌤PPT-18p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_6.py

import re

pattern = r"^abc"
text = ["abc", "abcd", "abc15", "dabc", "", "s"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )


print("---------------------------------")
