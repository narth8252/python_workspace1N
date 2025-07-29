# (myenv1) C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\머신러닝>C:/Users/Admin/.conda/envs/myenv1/python.exe c:/Users/Admin/Documents/GitHub/python_workspace1N/데이터분석7월/한글차트.py
# ['Malgun Gothic', 'Batang','Calibri', 'Noto Sans KR','Noto Serif KR', 'Malgun Gothic']

## 파이썬 차트 한글 폰트 설정과 정리(안그러면 끌어오고 복잡해짐)
# 한글이 잘 표시되는 matplotlib 차트를 만들기 위해 폰트 설정 방법을 체계적으로 정리합니다.

### 1. 내컴퓨터의 한글폰트 확인하기
# 파이썬 내장 라이브러리로 설치된 한글 폰트를 확인할 수 있습니다.
import matplotlib.font_manager as fm
font_list = [font.name for font in fm.fontManager.ttflist]
print(font_list)
# - 출력된 목록에서 'Malgun Gothic', 'Noto Sans KR', 'Noto Serif KR', 'Batang' 등 한글 지원 폰트를 확인할 수 있습니다.
# - 사용할 수 있는 폰트명을 그대로 사용하거나, 원하는 폰트(ttf) 파일 경로를 지정할 수 있습니다.

### 2. 차트 한글폰트 기본설정
# **추천 폰트**
# - PPT, 웹사이트, 일반문서: `Noto Sans KR`, `Malgun Gothic`
# - 보고서, 논문, 인쇄물: `Noto Serif KR`, `Batang` (명조계열, Serif체)

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic' # 내 컴퓨터에 설치된 폰트명 입력
plt.rcParams['axes.unicode_minus'] = False    # - 부호 깨짐 방지

# 차트 테스트
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species')
plt.suptitle('iris데이터셋 산포도행렬')
plt.show()

### 3. 폰트 직접경로 지정
# 내장 폰트가 없거나, 한글 깨짐이 해결되지 않는 경우 폰트파일(.ttf) 경로 지정 권장
from matplotlib import font_manager, rc
# 예시: 윈도우 기본경로
font_path = "c:/Windows/Fonts/malgun.ttf"

# 또는 자신의 작업폴더안에 폰트복사하여 상대경로 호출
# 예시: ../fonts/malgun.ttf (상위 폴더)
# font_path = "../fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

### 4. 폰트 상대경로 지정팁
# - 스크립트와 폰트 파일 위치에 따라 상대경로를 적절히 지정
#   - 예) `머신러닝/스크립트.py`, `../fonts/malgun.ttf`

### 5. 참고: 전체 코드 예시 (일목요연하게)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# [1] 폰트 경로 지정(절대/상대 모두 가능)
font_path = "c:/Windows/Fonts/malgun.ttf"  # 또는 "./fonts/malgun.ttf", "../fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()

plt.rcParams['font.family'] = font_name     # 폰트 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스(-) 깨짐 방지

iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species')
plt.suptitle('iris데이터셋 산포도행렬')
plt.show()

print("-------------------------------------------------")
#### ⚡️ 팁 요약
# - 설치된 폰트 확인: `matplotlib.font_manager.fontManager.ttflist`
# - 적합한 폰트명 선택 또는 ttf 경로 직접 지정
# - 마이너스 깨짐 방지: `plt.rcParams['axes.unicode_minus'] = False`
# - 폰트 파일 옮길 때 경로 정확히 지정
# - 긴글/보고서는 명조(Serif), 프리젠테이션/웹사이트는 고딕(Sans) 추천
# 실제 코드에 주석을 잘 달아두면 한글 차트 작성이 한결 쉬워집니다.

print("-------------------------------------------------")
#절대경로지정 사용법 권장
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
print("-------------------------------------------------")
#상대경로지정 - 내컴폰트를 작업폴더에 복붙(malgun.ttf은기본깔림)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\fonts
# 만약 폰트 파일이 위 경로에 있다면, 스크립트가 실행되는 위치에서 상대경로를 "../fonts/malgun.ttf" (한 단계 상위 폴더로 올라가기)로 지정해야 합니다.
# 즉, 스크립트가 "...\데이터분석7월\머신러닝" 폴더에 있고, 폰트가 "...\데이터분석7월\fonts"에 있다면,
#       스크립트위치	      폰트위치	                        상대경로지정
# ...데이터분석7월\머신러닝	...데이터분석7월\fonts\malgun.ttf	"../fonts/malgun.ttf"
# ...데이터분석7월	       ...데이터분석7월\fonts\malgun.ttf   "./fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname="../fonts/malgun.ttf").get_name()

print("-------------------------------------------------")
# 1. 한글 폰트 절대경로 지정 (윈도우 예시)
#절대경로지정 사용법 권장
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
print(font_name)

# 2. Seaborn 스타일 sns.set_style 세팅
#시본sns.set_style할때는 무조건rcParam위에 써야함. 안그러면 한글폰트깨짐
# seaborn에서 차트 스타일을 먼저 지정하는 것이 좋습니다.
# 스타일 지정 후 폰트 설정을 해야 폰트가 덮어써지지 않기 때문입니다
sns.set_style('whitegrid') #{darkgrid, whitegrid, dark, white, ticks}

plt.rcParams['font.family'] = font_name    #폰트설정
plt.rcParams['axes.unicode_minus'] = False #눈금에서 한글깨짐 문제발생

print("-------------------------------------------------")
# font_manager.FontProperties를 사용해 폰트 경로를 직접 지정하면, 해당 폰트가 운영체제에 설치되어 있지 않아도 파이썬 코드에서 해당 폰트를 사용할 수 있습니다. 
# 이 방식의 핵심은 폰트 파일(.ttf, .otf 등)을 특정 경로에 두고, 그 경로를 직접 지정해 불러오는 것입니다. 아래에 세부적인 동작 원리와 주의사항을 설명합니다.

# 어떻게 동작하나요?
# • 폰트 파일(.ttf 등)을 다운로드하여 원하는 폴더(예: 프로젝트 디렉토리, 임의 폴더 등)에 복사합니다. 반드시 C:\Windows\Fonts에 존재할 필요는 없습니다.
# • 예를 들어, C:/myfonts/NanumGothic.ttf와 같이 경로를 설정할 수 있습니다.
# • 코드에서 FontProperties(fname='경로/폰트명.ttf')와 같이 경로를 직접 지정하면, 
# matplotlib이 시스템 글로벌 폰트 목록이 아닌, 해당 위치의 폰트를 직접 불러와 적용합니다.
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "C:/myfonts/NanumGothic.ttf"  # 설치할 필요 없이, 이 경로에 파일만 있으면 됨!
font_name = font_manager.FontProperties(fname=font_path).get_name()  # 내부 폰트명 추출

plt.rcParams['font.family'] = font_name  # 폰트 전역 적용
# 또는 개별적으로 적용하려면 title 등에 fontproperties 인자를 직접 삽입
# • 이렇게 하면 시스템 폰트 등록 없이 해당 폰트만을 matplotlib에서 사용할 수 있습니다.