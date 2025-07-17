# 250716 PM2시 
# matplotlib와 seaborn 사용시 한글폰트 안깨지게 설정하고, 여러함수를 한글라벨로 시각화
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import os
from matplotlib import font_manager, rc

# 1. 한글 폰트 절대경로 지정 (윈도우 예시)
#절대경로지정 사용법 권장
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
print(font_name)

# 2. Seaborn 스타일 sns.set_style 세팅
#시본sns.set_style할때는 무조건rcParam위에 써야함. 안그러면 한글폰트깨짐
# seaborn에서 차트 스타일을 먼저 지정하는 것이 좋습니다.
# 스타일 지정 후 폰트 설정을 해야 폰트가 덮어써지지 않기 때문입니다
sns.set_style('whitegrid') #{darkgrid, whitegrid, dark, white, ticks}

# 3. 한글폰트/마이너스 깨짐 방지
plt.rcParams['font.family'] = font_name    #폰트설정
plt.rcParams['axes.unicode_minus'] = False #눈금에서 한글깨짐 문제발생

# 4. 히스토그램-분포도(통계학적으로 매우중요)
#loc: float = 
#scale: float
plt.plot(x, x, label='선형', color='g')  #green

plt.title('제목')
plt.xlabel('x축')
plt.ylabel('y축')
plt.legend(loc="best")
plt.show()