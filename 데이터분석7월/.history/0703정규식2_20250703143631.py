#쌤PPT-8p. 13회차_데이터셋_백현숙_0915_1차.pptx
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\정규식
# 파일명 : exam13_1.py
"""
◈ 우편번호 형식 맞추기 정규식 예제
우리나라의 우편번호는 6자리에서 5자리로 체계가 바뀌었습니다. 정수 값을 입력을 받아서 이 데이터가 우편번호 형식에 맞는지 확인해보는 예제를 만들어 보겠습니다. 
우편번호 패턴 방식 : \d{5}$   <- 정수 5개만 가능하다 

ㆍ정규식 패턴: \d{5}$
 - \d{5}: 숫자 5자리
 - $: 문자열 끝에 위치해야 함
ㆍ입력: 사용자에게 우편번호를 입력받음
ㆍ검사: re.match()로 입력값이 5자리 숫자인지 확인
ㆍ결과 출력: 일치하면 "매치합니다.", 아니면 "매치하지 않습니다." 출력
예시
ㆍ입력: 12345 → 매치합니다.
ㆍ입력: 1234a 또는 a12345 → 매치하지 않습니다.
참고
ㆍmatch()는 문자열의 시작부터 일치하는지 확인합니다.
ㆍ5자리 숫자만 허용하며, 그 외는 모두 불일치로 처리합니다
"""
import re

# 1. 정규식 패턴 정의
# 5자리 숫자(문자열 끝에 위치)만 허용하는 정규식 패턴을 만듭니다.
pattern = r'\d{5}$' #영문폰트라서 원화표시가 아닌 역슬래쉬로 보임
#$안붙이면 5자리 아니어도 보여짐.

# 2.사용자 입력 받기
zipcode = input("우편번호를 입력하세요") #231564dfgefs         
regex = re.compile(pattern)

#match함수가 패턴이 반드시 시작위치에 있어야한다. a12234
result = regex.match(zipcode) #일치하는 형식없으면 None 반환
print(result)
if result == None:
    print("매치하지 않습니다.")
else:
    print("매치합니다.")
