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
#\d?:숫자0~1개, \d+:숫자1개 이상, \d*:숫자0개이상 → 모두문자열의 맨앞에서 검사
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
패턴 \d?: ['abc-X', '1abc-O', '12abc-O', '123-O', 'aa12ab-X']
→ 숫자가 맨 앞에 0개 또는 1개 있으면 O
패턴 \d+: ['abc-X', '1abc-O', '12abc-O', '123-O', 'aa12ab-O']
→ 숫자가 맨 앞에 1개 이상 있으면 O
패턴 \d*: ['abc-X', '1abc-O', '12abc-O', '123-O', 'aa12ab-X']
→ 숫자가 맨 앞에 0개 이상(실제로는 빈 문자열도 매치되지만, 코드상 빈 문자열은 X로 처리)
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
"""
문제. 
사업자 번호는 앞3자리 가운데 2자리 뒤 5자리로 구성되어 있습니다. 문서에서 사업자 번호만 추출하여 그중 개인 사업자 관련 번호만 추출하고자 합니다. 사업자 번호의 각 자릿수의 의미는 다음과 같습니다.

사업자번호의 의미 : 000-00-00000
앞 3자리 - 관할세무서 번호 
가운데 2자리 - 사업자의 성격을 나타냄(개인사업자 : 90~99)
마지막 5자리 - 사업자용4자리+검증용1자리 
다음 데이터들로 부터 사업자 번호들을 추출하고 그 중에 개인 면세 사업자에 해당하는 사업자 번호만 출력하기 바랍니다
"""
사업자 번호는 앞3자리 가운데 2자리 뒤 5자리로 구성되어 있습니다. 문서에서 사업자 번호만 추출하여 그중 개인 사업자 관련 번호만 추출하고자 합니다. 사업자 번호의 각 자릿수의 의미는 다음과 같습니다.

사업자번호의 의미 : 000-00-00000
앞 3자리 - 관할세무서 번호 
가운데 2자리 - 사업자의 성격을 나타냄(개인사업자 : 90~99)
마지막 5자리 - 사업자용4자리+검증용1자리 
다음 데이터들로 부터 사업자 번호들을 추출하고 그 중에 개인 면세 사업자에 해당하는 사업자 번호만 출력하기 바랍니다
"""
contents = """
    우리커피숍 100-90-12345
    영풍문고 101-91-12121
    영미청과 102-92-23451
    황금코인 103-89-13579
    우리문구 104-91-24689
    옆집회사 105-82-12345
contents = """
    우리커피숍 100-90-12345
    영풍문고 101-91-12121
    영미청과 102-92-23451
    황금코인 103-89-13579
    우리문구 104-91-24689
    옆집회사 105-82-12345
"""

