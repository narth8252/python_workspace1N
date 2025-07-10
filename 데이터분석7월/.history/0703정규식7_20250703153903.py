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
"""
abc - O
abcd - O
abc15 - O
dabc - X
 - X
s - X
"""
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
"""
abc - O
dabcd - X
asdabc - O
d12abc - O
 - X
s - X
"""
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
"""
python - O
Python - O
PYTHON - X
"""
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
"""
python - X
Python - O
PYTHON - O
korea - X
KOREA - O
Korea - O
"""
print("---------------------------------")

# 5. \d? \d+ \d* : 숫자 패턴 실험
# \d? : 숫자가 0개 또는 1개 있는지 검사(있어도 되고, 없어도 됨. 가장 앞에서 한 글자만 검사)
# \d+ : 숫자가 1개 이상 연속해서 있는지 검사(최소 1개 이상의 숫자가 필요)
# \d* : 숫자가 0개 이상 있는지 검사(숫자가 없어도 되고, 여러 개여도 됨)
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
패턴 \d?: 
['abc-X', '1abc-O', '12abc-O', '123-O', 'aa12ab-X']
패턴 \d+: ['abc-X', '1abc-O', '12abc-O', '123-O', 'aa12ab-O']
패턴 \d*: ['abc-X', '1abc-O', '12abc-O', '123-O', 'aa12ab-X']
"""
print("---------------------------------")

# 6. [^문자들] : 대괄호 안의 문자들을 제외한 문자 찾기
pattern = r"[^a-z]"  # 소문자 알파벳이 아닌 문자 찾기
text = ["python", "Python3", "123", "hello!", "KOREA", " "]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item)
    if result:
        print(f"{item} - '{result.group()}' - O")
    else:
        print(f"{item} - X")
"""
python - X
Python3 - '3' - O
123 - '1' - O
hello! - '!' - O
KOREA - 'K' - O
  - ' ' - O
"""
print("---------------------------------")