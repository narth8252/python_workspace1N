from konlpy.tag import Kkma
from konlpy.utils import pprint

msg = """
오픈소스를 이용하여 형태소 분석을 배워봅시다. 형태소 분석을 지원하는 라이브러리가 많습니다. 
각자 어떻게 분석하지는 살펴보겠습니다. 
이건 Kkma모듈입니다.
"""
kkma = Kkma()
print(kkma.sentences(msg))      # 문장 분석
print(kkma.morphs(msg))         # 형태소 분석
print(kkma.nouns(msg))          # 명사
print(kkma.pos(msg))            # 품사태깅

#시스템환경변수
#JAVA_HOME
# C:\Program Files\java\openjdk-11+28_windows-x64_bin\jdk-11

# JAVA_HOME
# C:\Program Files\java\openjdk-11+28_windows-x64_bin\jdk-11
# C:\Program Files\ojdkbuild\java-11-openjdk-11.0.15-1
# echo %JAVA_HOME%
# 파워셸에서는
# echo $env:JAVA_HOME
# 치면 나옵니다
#환경변수 - 시스템변수 - Path>새로만들기 - C:\Program Files\ojdkbuild\java-11-openjdk-11.0.15-1\bin (맨위로이동)
#VS코드, 컴퓨터 껐다켜기.

# cmd 관리자권한실행
# pip install pytagcloud 
# pip install pygame 
# pip install WordCloud

# VS코드 cmd창
# (base) C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝>conda activate myenv1
# (myenv1) C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝>python 0704형태소.py 