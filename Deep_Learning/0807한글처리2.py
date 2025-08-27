#정규식 사용해서 쓸데없는 문자 제거하기 : 영어의 구두점 제거
from konlpy.tag import Okt
import re #정규식 모듈

"""
re.sub(r"패턴", "대체할문자열", "내용", count=0)
패턴 : 찾을 정규식 패턴
"""

def clean_text(text):
    text = text.lower() # 대문자 → 소문자
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
    return text
    #r → escape문자무력화(\s공백, \n줄바꿈) 패턴에 \ 직접사용할경우 escape문자로 인식하면안된다
    #[]필요한 문자열 묶기, [^]제외하고, ^[] ~로 시작하는
    #[^가-힣] : 한글빼고 나머지를 제외한다.멀쩡한 완성형한글만 인식(ㅋㅋ 이런거 제외)

text = "ㅋㅋ AI는 인간의 일을 대체하게 될겁니다. 장점도 있고 단점도 있습니다. 인간의 반복적이고 힘든일을 대체했으면~!"
text = clean_text(text)
print(text)
okt = Okt()
print(okt.morphs(text))


