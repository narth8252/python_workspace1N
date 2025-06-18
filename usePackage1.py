#0513 1:30pm 모듈설명후 패키지 실습 3-3
#신규폴더 mypackage1만들고 그 아래 방금만든 모듈.py 파일2개 넣어놓고 실행하면 가져와서 출력됨
#이거useModule.py는 workspace1N폴더에 있어야하고 나머지 __init__.py이랑 복붙한mymodule2.py랑 는 새로만든 mypackage1폴더에 있어야함.
from mypackage1.mymodule2 import isEven, toUpper
print( isEven(10)) #True
print( toUpper("Korea")) #KOREA

"""
(base) PS c:/Users/Administrator/Documents/python_workspace1N/usePackage1.py
여기가 패키지 시작입니다.
True
KOREA
"""