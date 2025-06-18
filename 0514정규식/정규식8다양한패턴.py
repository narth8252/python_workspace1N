#쌤ppt 22~p  0514_11:40am
#
#젤많이쓰는거 없어와서 만드신건데 gpt한테 달라고하면 됨.
#gpt검색어: 파이썬 정규식 이메일 패턴

import re

pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
# 설명:
# ^ / $: 문자열의 시작과 끝
# [a-zA-Z0-9._%+-]+: 사용자 이름 (username)
# @: 반드시 포함되는 @ 기호
# [a-zA-Z0-9.-]+: 도메인 이름
# \.[a-zA-Z]{2,}: 최상위 도메인 (예: .com, .net, .co.kr 등)

# ◈정규식 패턴 표현
# 패턴 : ^  
# 설명 : 시작이 이 패턴으로 이루어져야 한다 
# 예제 : 패턴이 ^abc 라면 문자열의 시작이 abc이어야 한다 
#    abcde, abc, abc123 다 된다

# 패턴 : $ 
# 설명 : 이 패턴으로 끝나야 한다 
# 예제 : 패턴이 abc$라면 문자열의 끝은 abc여야 한다, match 함수는 안된다. 
# abc, dabc, 123abc 다 된다 

# 패턴 : [문자들]
# 설명 : []에 속한 문자들만 해당된다 가능한 문자열의 집합을 의미한다 
# 예제 :패턴이 [p|P]ython 라면
# python, Python 은 가능하나 , PYTHON은 안된다. 
# [A-Z] 첫글자가 알파벳 대문자만 가능하다 
# KOREA-O, korea-X, Korea-O


# 패턴 : [^문자들]  
# 설명 : 피해야할 문자들의 집합이다. 이 문자들로 구성된 단어만 피한다
# 예제: 패턴이 [^abc] 라면 
#    a, b, c, abc, acb, bca – x, d-o

# 패턴 : | 
# 설명 : or 연산 둘중 하나만 일치하면 된다. 
# 예제) 패턴이 [k|K]orea라면    
# korea, Korea – o, Corea - x

# 패턴 : ?, +, *
# 설명 : ? 앞의 패턴이 없거나 하나이어야 한다 
#           + 앞의 패턴이 하나 이상 있어야 한다 
#           * 앞의 패턴이 0개 이상이어야 함, 반복을 의미 
# 예제 : \d?  - 숫자가 없거나 하나만 있어야 한다 
#           \d+  - 숫자가 하나 이상이어야 함 
#           \d*  - 숫자가 없거나 하나 이상이어야 함 



# ◈정규식 패턴 표현예제 
# 파일명 : exam13_7.py
import re

# pattern = r"^abc"
# pattern = r"abc"
pattern = r"^abc$"
text = ["abc", "abcd", "abc15", "dabc", "", "s"]
repattern = re.compile(pattern)

for item in text:
    result = repattern.search(item)
    if result:
        print(item, "- O" )
    else:
        print(item, "- X" )
"""pattern = r"^abc"   pattern = r"abc"    pattern = r"^abc$"
abc - O                abc - O             abc - O
abcd - O               abcd - O            abcd - X
abc15 - O              abc15 - O           abc15 - X
dabc - X               dabc - O            dabc - X
 - X                    - X                 - X      
s - X                  s - X               s - X
"""

"""pattern = r"abc"
abc - O
abcd - O
abc15 - O
dabc - O
 - X
s - X
"""

"""pattern = r"^abc$"
abc - O
abcd - X
abc15 - X
dabc - X
 - X
s - X
"""

