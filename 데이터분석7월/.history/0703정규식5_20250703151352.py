

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
