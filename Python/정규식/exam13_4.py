import re
#전화번호만 추출하기
text = """
    phone : 010-0000-0000 email:test1@nate.com
    phone : 010-1111-1111 email:test2@naver.com
    phone : 010-2222-2222 email:test3@gmail.com
    """
print()
print("--- 전화번호 추출하기 ---")
phonepattern = r"\d{2,3}-\d{4}-\d{4}"

matchObj = re.findall( phonepattern, text) #string 으로 보낸다. 

for item in matchObj:
    print( item)

print("--- 이메일 추출하기 ---") 
emailpattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b"

matchObj = re.findall( emailpattern, text)
for item in matchObj:
    print( item)
print()
