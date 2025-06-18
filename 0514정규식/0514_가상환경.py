#0514 2시pm
#위키독스 Python강좌와통계2.5.가상환경다루기 https://wikidocs.net/252578
#윈도우랑 리눅스 os랑 가상환경만드는 방법 다름.
# 가상환경만들기  → 강화학습: 게임에 적용되는 머신러닝 알고리즘(파이썬 3.9이하만)
#프로젝트별로 별도의 환경을 만들수있다.
#파이썬에서 가상환경(virtual environment)은 프로젝트마다 독립된 파이썬 환경을 제공하는 시스템입니다. 
# 가상환경은 각 프로젝트에 필요한 라이브러리와 패키지를 다른 프로젝트와 독립적으로 설치하고 관리할 수 있게 해줍니다. 
# 이를 통해 프로젝트 간의 종속성 충돌을 방지하고, 서로 다른 프로젝트가 서로 다른 라이브러리 버전을 요구하는 경우에도 문제가 발생하지 않습니다.
#의존성관리 
conda create --name 가상환경명 python=3.8
conda create --name myenv1 python=3.8 
#myenv1이라는 가상환경을 만들고 파이썬3.8을 설치한다.

#1.cmd 치고 >명령프롬프트나 Anaconda Prompt를 우클릭>관리자권한으로 실행
#2.(base) C:\Users\Administrator>가 있으면 "conda create --name myenv1 python=3.8"치면됨.
"""
(base) C:\Users\Administrator>conda create --name myenv1 python=3.8
Channels:
 - defaults
Platform: win-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: C:\ProgramData\anaconda3\envs\myenv1

  added / updated specs:
    - python=3.8


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    pip-24.2                   |   py38haa95532_0         2.4 MB
    python-3.8.20              |       h8205438_0        19.4 MB
    setuptools-75.1.0          |   py38haa95532_0         1.6 MB
    vc-14.42                   |       haa95532_5          11 KB
    vs2015_runtime-14.42.34433 |       hbfb602d_5         1.2 MB
    wheel-0.44.0               |   py38haa95532_0         137 KB
    ------------------------------------------------------------
                                           Total:        24.7 MB

The following NEW packages will be INSTALLED:

  ca-certificates    pkgs/main/win-64::ca-certificates-2025.2.25-haa95532_0
  libffi             pkgs/main/win-64::libffi-3.4.4-hd77b12b_1
  openssl            pkgs/main/win-64::openssl-3.0.16-h3f729d1_0
  pip                pkgs/main/win-64::pip-24.2-py38haa95532_0
  python             pkgs/main/win-64::python-3.8.20-h8205438_0
  setuptools         pkgs/main/win-64::setuptools-75.1.0-py38haa95532_0
  sqlite             pkgs/main/win-64::sqlite-3.45.3-h2bbff1b_0
  vc                 pkgs/main/win-64::vc-14.42-haa95532_5
  vs2015_runtime     pkgs/main/win-64::vs2015_runtime-14.42.34433-hbfb602d_5
  wheel              pkgs/main/win-64::wheel-0.44.0-py38haa95532_0


Proceed ([y]/n)?
"""

#3.이거뜨면 y
#나중에 웹,머신러닝 뭐 왔다갔다하며 해야돼서 다 각각 가상환경 깔아서 사용.
# 3.8밑으로 깔아야됨.
"""
Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate myenv1
#
# To deactivate an active environment, use
#
#     $ conda deactivate


(base) C:\Users\Administrator>
"""
#4.이거뜨면 clear cls 쳐
"""
(base) C:\Users\Administrator>clear cls
'clear'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는
배치 파일이 아닙니다.
(base) C:\Users\Administrator>
"""
#5.conda activate myenv1 쳐.
# (myenv1) C:\Users\Administrator>

#6.python --version 쳐
# (myenv1) C:\Users\Administrator>python --version
# Python 3.8.20

#7.exit() 치고 빠져나와.

#8.(myenv1) C:\Users\Administrator> pip install numpy
# Collecting numpy
#   Downloading numpy-1.24.4-cp38-cp38-win_amd64.whl.metadata (5.6 kB)
# Downloading numpy-1.24.4-cp38-cp38-win_amd64.whl (14.9 MB)
#    ---------------------------------------- 14.9/14.9 MB 29.2 MB/s eta 0:00:00
# Installing collected packages: numpy
# Successfully installed numpy-1.24.4
#(myenv1) C:\Users\Administrator>

#10. 가상환경(myenv1) 설치된데 출력확인.
# 파이썬 프롬프트창 출력시
# hello.py(內명령어 print("Hello"))파일 만들고 vscode 프롬프트 출력창에 conda activate myenv1쳐
# 위치가 (base)  → (myenv1)로 변경되면
# 방금만든 python hello.py을 프롬프트창에 치면 Hello 출력됨.
# (base) PS C:\Users\Administrator\Documents\python_workspace1N> conda activate myenv1
# (myenv1) PS C:\Users\Administrator\Documents\python_workspace1N> python hello.py
# Hello
# (myenv1) PS C:\Users\Administrator\Documents\python_workspace1N> 
#가상환경끄고싶으면 deactivate 쳐

#11.Anaconda Prompt창에 conda deactivate 치면 가상환경 끔 (conda 생략가능)
#(myenv1) C:\Users\Administrator>conda deactivate
#(base) C:\Users\Administrator>

#12.가상환경만들어진 위치확인 C:\ProgramData\anaconda3\envs\myenv1  이위치에 만들어짐.
#폴더들어가서 보면 파이썬 기존설치랑 비슷하게 다 설치돼있음.

#13.가상환경 삭제하고싶으면 폴더삭제
#C:\ProgramData\anaconda3\envs\myenv1 내가 만든 요폴더.
#C:\ProgramData\anaconda3\envs여기를 설치하면 안됨!