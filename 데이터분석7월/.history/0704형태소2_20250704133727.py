from konlpy.tag import Hannanum
from konlpy.utils import pprint

msg = """
오픈소스를 이용하여 형태소 분석을 배워봅시다. 형태소 분석을 지원하는 라이브러리가 많습니다. 
각자 어떻게 분석하지는 살펴보겠습니다. 
이건 Hannanum모듈입니다.
"""
hannanum = Hannanum()
print(hannanum.analyze(msg))      # 문장 분석
print(hannanum.morphs(msg))         # 형태소 분석
print(hannanum.nouns(msg))          # 명사
print(hannanum.pos(msg))            # 품사태깅

from konlpy.tag import Okt
twitter = 
#시스템환경변수
#JAVA_HOME
# C:\Program Files\java\openjdk-11+28_windows-x64_bin\jdk-11