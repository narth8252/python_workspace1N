#쌤PPT-20p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_6.py

import re

pattern = r"^abc"
text = ["abc", "abcd", "abc15", "dabc", "", "s"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O")
    else:
        print(item, "- X")

print("---------------------------------")
pattern = r"abc$"
text = ["abc", "dabcd", "asdabc", "d12abc", "", "s"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item) 
        if result:
        print(item, "- O" )
    else:
        print(item, "- X" )
print("---------------------------------")
pattern = r"[p|P]ython"
text = ["python", "Python", "PYTHON"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item) 
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )
pattern = r"[A-Z]"
text = ["python", "Python", "PYTHON", "korea", "KOREA", "Korea"]
repattern = re.compile(pattern)for item in text:
    result = repattern.search(item) 
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )

patterns=[r"\d?", r"\d+", r"\d*"]
text = ["abc", "1abc", "12abc", "123", "aa12ab"]
for pattern in patterns:
    resultList=[]
    for item in text:
        result = re.search(pattern, item)
        if result == None:
            resultList.append(item+"-X")
        else:
            resultList.append(item+"-O")
    print(resultList)
