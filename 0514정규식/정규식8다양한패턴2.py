#0514 1시 pm
#외우라고 공부하는것 아니고 알아놓고 gpt한테 물어봐서 갖다 쓰라고 공부하는 것임.
# 쌤ppt 28p ◈정규식 패턴 표현예제 
# 파일명 : exam13_9.py
import re

# python 또는 Python 문자열을 찾습니다. 그러나 PYTHON은 대문자로만 구성되어 있으므로, 이 문자열은 패턴을 만족하지 않습니다.
#문자열에서 대문자가 하나라도 포함되어 있는지 확인합니다. 만약 문자열에 대문자가 있으면 "- O"를, 없으면 "- X"를 출력합니다.
pattern = r"[p|P]ython"
text = ["python", "Python", "PYTHON"]
repattern = re.compile(pattern)

for item in text:
    result = repattern.search(item) 
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )

pattern = r"[A-Z]" #대문자 하나라도 포함되면 출력
text = ["python", "Python", "PYTHON", "korea", "KOREA", "Korea"]
repattern = re.compile(pattern)
for item in text:
    result = repattern.search(item) 
    #match함수는 첫시작만을 보기때문에 적용안됨
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )
"""pattern = r"[p|P]ython"
python - O
Python - O
PYTHON - X
python - X
Python - O
PYTHON - O
korea - X
KOREA - O
Korea - O

    re.match()과 re.search()의 차이
re.match()는 문자열의 시작에서만 패턴을 찾습니다. 따라서 re.match(r"[A-Z]", "korea")는 None을 반환합니다.
re.search()는 문자열의 어디에서나 패턴을 찾습니다. 따라서 re.search(r"[A-Z]", "korea")는 None을 반환하지만, re.search(r"[A-Z]", "Korea")는 대문자 "K"를 찾습니다.
따라서 대문자가 문자열의 시작이 아닐 수도 있으므로, re.search()를 사용하는 것이 더 적합합니다.

[p|P]ython 패턴은 python과 Python을 찾기 위해 사용됩니다. 그러나 PYTHON은 이 패턴을 만족하지 않습니다. 만약 모든 대소문자 형태를 포함하려면 r"[Pp]ython" 대신 r"(?i)python"를 사용할 수 있습니다. (?i)는 패턴이 대소문자 구분 없이 일치하도록 합니다.
이 두 가지 예제를 통해 정규식 패턴을 사용하여 문자열에서 특정 조건을 만족하는 문자열을 찾는 방법을 이해할 수 있으실 것입니다. 도움이 되셨길 바랍니다. 추가적인 질문이 있으시면 언제든지 물어봐 주세요.
"""


# 28p ◈정규식 패턴 표현예제, 파일명 : exam13_10.py
import re

#패턴을3개만들어놓고 이중for 루프로 돌린것임.
# ? 정수가 있어도 그만 없어도그만, 있을수도있고 없을수도있고.
# + 정수가 하나이상포함
# * 정수있거나없거나 상관x
patterns=[r"\d?", r"\d+", r"\d*"]
text = ["abc", "1abc", "12abc", "123", "aa12ab"]

# 각 패턴에 대해 반복
for pattern in patterns:
    resultList=[]

    # 각 텍스트 항목에 대해 반복
    for item in text:
        result = re.search(pattern, item)

        # 숫자가 발견되지 않으면 "-X"를 추가
        if result == None:
            resultList.append(item+"-X")
        # 숫자가 발견되면 "-O"를 추가
        else:
            resultList.append(item+"-O")

    print(resultList)
"""
['abc-O', '1abc-O', '12abc-O', '123-O', 'aa12ab-O']
['abc-X', '1abc-O', '12abc-O', '123-O', 'aa12ab-O']
['abc-O', '1abc-O', '12abc-O', '123-O', 'aa12ab-O']
 결과 설명
    패턴 1: r"\d?"
?는 0번이나 1번 등장하는 것을 의미합니다. 따라서 숫자가 있으면 "-O", 없으면 "-X"가 추가됩니다.
    패턴 2: r"\d+"
+는 1번 이상 등장하는 것을 의미합니다. 따라서 하나 이상의 숫자가 있으면 "-O", 없으면 "-X"가 추가됩니다.
    패턴 3: r"\d*"
*는 0번 이상 등장하는 것을 의미합니다. 따라서 숫자가 없거나 하나 이상 있으면 "-O", 숫자가 전혀 없는 경우에도 "-O"가 추가됩니다.

 예시 결과
    패턴 1: r"\d?"
["abc-X", "1abc-O", "12abc-O", "123-O", "aa12ab-O"]
    패턴 2: r"\d+"
["abc-X", "1abc-O", "12abc-O", "123-O", "aa12ab-O"]
    패턴 3: r"\d*"
["abc-O", "1abc-O", "12abc-O", "123-O", "aa12ab-O"]
"""

# 29p  패턴 표현예제, 파일명 : exam13_10.py
import re

patterns=[r"\d?", r"\d+", r"\d*"]
text = ["abc", "1abc", "12abc", "123", "aa12ab"]

for pattern in patterns:
    resultList=[]
    for item in text:
        result = re.search(pattern, item)
        if result == None:
            resultList.append(item+"-X")
        else:
            resultList.append(item+"-O")

    print(resultList)

#0514 1:30pm 30p◈정규식 그룹화-파일명 : exam13_11.py
#정규식 패턴을 사용하여 전화번호를 추출하고 그룹화하는 방법
#Q.전화번호를 세 가지 그룹으로 나누어 추출하는 방법을 보여줍니다.
import re
# 전화번호 패턴을 괄호로 그룹화
contents = "문의사항이 있으면 010-1234-6789 으로 연락주시기 바랍니다."

#전화번호 패턴을 괄호를 이용해 그룹화 한다
#3개의 구룹으로 나누어졌으므로 group(1), group(2), group(3)로 각각의 번호를 추출
#1.그룹화:전화번호를 3부분으로 나눠 추출
# (\d{3}): 1번째 그룹으로, 3자리 숫자를 나타냅니다. 예: 010
# (\d{4}): 2,3번째 그룹으로, 각각 4자리 숫자를 나타냅니다. 예: 1234, 6789
pattern = r'(\d{3})-(\d{4})-(\d{4})'

#group() 함수: re.search()가 성공하면, group() 함수를 사용하여 각 그룹의 값을 추출할 수 있습니다.
regex = re.compile(pattern)
result = regex.search(contents)
if result != None:
    phone1 = result.group(1) # 첫번째 그룹: 010
    phone2 = result.group(2) # 두번째 그룹: 1234
    phone3 = result.group(3) # 세번째 그룹: 6789
    print(phone1)
    print(phone2)
    print(phone3) 
else:
    print("전화번호가 없습니다.")
"""
010
1234
6789
"""


"""
    그룹화: 정규식 패턴에서 ()를 사용하여 그룹을 생성할 수 있습니다. 이 예제에서는 전화번호를 세 부분으로 나누어 추출하는 데 사용됩니다.
-(\d{3}): 첫 번째 그룹으로, 3자리 숫자를 나타냅니다. 예: 010
-(\d{4}): 두 번째와 세 번째 그룹으로, 각각 4자리 숫자를 나타냅니다. 예: 1234, 6789
    group() 함수: re.search()가 성공하면, group() 함수를 사용하여 각 그룹의 값을 추출할 수 있습니다.
-group(1): 첫 번째 그룹의 값을 반환합니다.
-group(2): 두 번째 그룹의 값을 반환합니다.
-group(3): 세 번째 그룹의 값을 반환합니다.
-group(0) 또는 group(): 전체 일치된 문자열을 반환합니다. 예: 010-1234-6789
    
-이 패턴은 한국의 유선전화 및 휴대전화 번호를 추출하는 데 적합합니다. 그러나 국제 전화번호나 다른 형식의 번호는 별도의 패턴이 필요할 수 있습니다.
-re.search() 함수는 문자열의 어디에서나 패턴을 찾습니다. 따라서 전화번호가 문자열의 시작에 위치하지 않아도 찾을 수 있습니다.
이 코드를 통해 전화번호를 그룹화하고 추출하는 방법을 이해할 수 있으실 것
"""

# ppt-32p ◈ 적용하기
# 사업자 번호는 앞3자리 가운데 2자리 뒤 5자리로 구성되어 있습니다. 문서에서 사업자 번호만 추출하여 그중 개인 사업자 관련 번호만 추출하고자 합니다. 사업자 번호의 각 자릿수의 의미는 다음과 같습니다.  
  
# 사업자번호의 의미 : 000-00-00000
#   앞 3자리 - 관할세무서 번호 
#   가운데 2자리 - 사업자의 성격을 나타냄(개인사업자 : 90~99)
#   마지막 5자리 - 사업자용4자리+검증용1자리(마지막1자리를 checkdigit이라고함.)
#   다음 데이터들로 부터 사업자 번호들을 추출하고 그 중에 개인 면세 사업자에 해당하는 사업자 번호만 출력하기 바랍니다 
  
import re
 
contents = """
    우리커피숍 100-90-12345
    영풍문고 101-91-12121
    영미청과 102-92-23451
    황금코인 103-89-13579
    우리문구 104-91-24689
    옆집회사 105-82-12345
"""
#데이터를 분리해야 하므로 그룹을 이용해서 패턴을 지정함
#  r'(\d{3})-(\d{2})-(\d{5})’
# 모든 요소를 분리해야 하므로 finditer  함수를 이용해서 분리하였음 
# 반환값인 MatchObject 객체의 group함수를 이용해서 데이터를 추출함
# 0번에는 사업자 번호가 전체가 들어있고 1번에는 앞 3자리 2번에는 가운데 2자리, 3에는 나머지 값이 있다. 
# group(2)를 추출하여 90~99 사이의 값인지 확인해야 하므로 str값을 int 형으로 전환하여 비교하여 그 사이에 존재하면 출력함  
#       (\d{3})1번그룹의 3자리숫자  
#       (\d{2})2번그룹의 2자리숫자 ,이 값이 90~99사이
#       (\d{5})3번그룹의 5자리 숫자
pattern = r'(\d{3})-(\d{2})-(\d{5})'
regex = re.compile(pattern)
result = regex.finditer(contents)
#finditer() 함수: 이 함수는 contents 문자열에서 패턴과 일치하는 모든 부분을 찾고, 각 MatchObject를 반환
print()

#group(2)를 추출하여 90~99 사이의 값인지 확인함
#group() 함수: MatchObject의 group() 함수를 사용하여 각 그룹의 값을 추출
# group(0): 전체 일치된 문자열을 반환합니다.
# group(1): 첫 번째 그룹의 값을 반환합니다.
# group(2): 두 번째 그룹의 값을 반환합니다.
# group(3): 세 번째 그룹의 값을 반환합니다.

#출력조건
#if init(item.group(2) >= 90 and int(item.group(2)<=99:는 
# 2번그룹의 값이 90~99사이인지 확인하고, 맞으면 출력
for item in result:
    if int(item.group(2))>=90 and int(item.group(2))<=99:
        print( item.group() )

print()
"""# 90~99 사이의 두 번째 그룹 값을 가진 사업자등록번호를 출력
100-90-12345
101-91-12121
102-92-23451
104-91-24689
"""
