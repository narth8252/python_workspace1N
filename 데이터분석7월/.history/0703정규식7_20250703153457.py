#쌤PPT-20p. 13회차_데이터셋_백현숙_0915_1차.pptx
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_6.py

import re

# 1. ^abc : abc로 시작하는 문자열 찾기
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

# 2. abc$ : abc로 끝나는 문자열 찾기
pattern = r"abc$"
text = ["abc", "dabcd", "asdabc", "d12abc", "", "s"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O")
    else:
        print(item, "- X")

print("---------------------------------")

# 3. [p|P]ython : python 또는 Python 찾기 (대소문자 구분)
pattern = r"[pP]ython"
text = ["python", "Python", "PYTHON"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O")
    else:
        print(item, "- X")

print("---------------------------------")

# 4. [A-Z] : 대문자 알파벳이 포함된 문자열 찾기
pattern = r"[A-Z]"
text = ["python", "Python", "PYTHON", "korea", "KOREA", "Korea"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O")
    else:
        print(item, "- X")

print("---------------------------------")

# 5. \d? \d+ \d* : 숫자 패턴 실험
patterns = [r"\d?", r"\d+", r"\d*"]
text = ["abc", "1abc", "12abc", "123", "aa12ab"]

for pattern in patterns:
    resultList = []
    for item in text:
        result = re.search(pattern, item)
        if result is None or result.group() == "":
            resultList.append(item + "-X")
        else:
            resultList.append(item + "-O")
    print(f"패턴 {pattern}: {resultList}")
"""
"""
print("---------------------------------")