#쌤PPT-11p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_1.py
"""
이 코드는 이메일 정규식을 작성 중인데,
네가 원하는 건 전화번호만 추출하는 코드야.

아래처럼 전체를 완성해줄게.
(이미 re 모듈은 임포트 돼 있고, 여러 사람의 전화번호와 이메일이 포함된 문자열에서 전화번호만 추출하는 코드)
"""

import re

# 텍스트 데이터
text = """
홍길동 phone : 010-0000-0000 email:test1@nate.com
임걱정 phone : 010-1111-1111 email:test2@naver.com
장길산 phone : 010-2222-2222 email:test3@gmail.com
"""

# 각 줄마다 이름, 전화번호, 이메일 추출
pattern = r'(\w+)\s+phone\s*:\s*(\d{3}-\d{4}-\d{4})\s+email:([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'

matches = re.findall(pattern, text)

# 딕셔너리 형태로 정리
result = []
for name, phone, email in matches:
    result.append({
        '이름': name,
        '전화번호': phone,
        '이메일': email
    })

# 출력
for item in result:
    print(item)

import re
# 전화번호만 추출하기
text = """
홍길동 phone : 010-0000-0000 email:test1@nate.com
임걱정 phone : 010-1111-1111 email:test2@naver.com
장길산 phone : 010-2222-2222 email:test3@gmail.com
"""

# ^ → 시작, $ → 끝, \b → 경계
# 이메일 정규표현식 (도메인 끝부분 2~4글자 제한)
pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$'
pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b"

matchObject = re.findall(pattern, text)
print(matchObject)


# 전화번호 정규식 패턴
# \d{3}: 숫자 3자리 (예: 010)
# -: 하이픈
# \d{4}: 숫자 4자리
# \b: 단어 경계 (전화번호 외의 문자열과 붙어 있을 경우 방지)
pattern = r"\b\d{3}-\d{4}-\d{4}\b"
phonepattern = r"\b\d{2,3}-\d{3,4}-\d{4}\b"
matchObject = re.findall(phonepattern, text)
print(matchObject)


matchObjects = re.findall(phonepattern, text)
print(matchObject)