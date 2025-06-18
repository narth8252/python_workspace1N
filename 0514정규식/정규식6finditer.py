
#18p finditer함수 _ 11:30 am
# finditer(pattern, string, flag=0)
# 문자열에 패턴과 매칭괴는 부분을 string리스트로 반환
# ex. 전화번호나 이메일만 추출 
#전화번호 체크패턴  phonepattern = r"\d{3}-\d{4}-\d{4}"
# 공백 문자를 고려한 패턴          r"\d{3}\s*-\s*\d{4}-\d{4}"
#이메일 체크패턴 r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b"
#emailpattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b"


#쌤ppt 16p
#finditer 함수 예제

import re
text = """
phone : 010-0000-0000 email: mail@nate.com
phone : 010-1111-1111 email: mail@naver.com
phone : 010-2222-2222 email: mail@gmail.com
phone : 02-123-4567 email: madkelil@fdklg.kdk
"""

print()

print("---------전화번호 추출하기-------")
#                 2,3자리   3,4자리 
phonepattern = r"\d{2,3}-\d{3,4}-\d{4}" 
matchObj = re.finditer(phonepattern, text)

for item in matchObj:
    print(item.group())
    print(item.span())
print()

print("---------이메일 추출하기-------")
emailpattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b"

matchObj = re.finditer(emailpattern, text)
for item in matchObj:
    print(item)
print()
