import re

pattern = r"[p|P]ython"
text = ["python", "Python", "PYTHON"]
repattern = re.compile(pattern)

for item in text:
    result = repattern.search(item) 
    #match함수는 첫 시작만을 보기때문에 적용안됨
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )

pattern = r"[A-Z]" #대문자가 하나라도 포함되면 
text = ["python", "Python", "PYTHON", "korea", "KOREA", "Korea"]
repattern = re.compile(pattern)

for item in text:
    result = repattern.search(item) 
    #match함수는 첫 시작만을 보기때문에 적용안됨
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )