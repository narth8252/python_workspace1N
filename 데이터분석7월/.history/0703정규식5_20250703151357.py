"""
이 코드는 이메일 정규식을 작성 중인데,
네가 원하는 건 전화번호만 추출하는 코드야.

아래처럼 전체를 완성해줄게.
(이미 re 모듈은 임포트 돼 있고, 여러 사람의 전화번호와 이메일이 포함된 문자열에서 전화번호만 추출하는 코드)
"""

import re

# 전화번호만 추출하기
text = """
홍길동 phone : 010-0000-0000 email:test1@nate.com
임걱정 phone : 010-1111-1111 email:test2@naver.com
장길산 phone : 010-2222-2222 email:test3@gmail.com
"""

# 전화번호 정규식 패턴
pattern = r"\b\d{3}-\d{4}-\d{4}\b"

# 추출
phone_numbers = re.findall(pattern, text)

# 출력
for num in phone_numbers:
    print(num)
