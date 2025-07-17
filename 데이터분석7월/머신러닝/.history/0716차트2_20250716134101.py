import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns

import os
from matplotlib import font_manager, rc
#font_manager - 폰트를 폰트객체화 
#rc - 폰트를 지정할 영역 (차트영역)

#절대경로지정 사용법 권장
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()

#상대경로지정 - 내컴폰트를 작업폴더에 복붙(malgun.ttf은기본깔림)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\fonts
# 만약 폰트 파일이 위 경로에 있다면, 스크립트가 실행되는 위치에서 상대경로를 "../fonts/malgun.ttf" (한 단계 상위 폴더로 올라가기)로 지정해야 합니다.
# 즉, 스크립트가 "...\데이터분석7월\머신러닝" 폴더에 있고, 폰트가 "...\데이터분석7월\fonts"에 있다면,
#       스크립트위치	      폰트위치	                        상대경로지정
# ...데이터분석7월\머신러닝	...데이터분석7월\fonts\malgun.ttf	"../fonts/malgun.ttf"
# ...데이터분석7월	       ...데이터분석7월\fonts\malgun.ttf   "./fonts/malgun.ttf"
# font_name = font_manager.FontProperties(fname="./fonts/malgun.ttf").get_name()
font_name = font_manager.FontProperties(fname="../fonts/malgun.ttf").get_name()
# rc('font', family=font_name)
print(font_name)
#시본셋

plt.rcParams['font.family'] = font_name    #폰트설정
plt.rcParams['axes.unicode_minus'] = False #눈금에서 한글깨짐 문제발생

x = np.linspace(0, 2, 100) #구간나누기. 0~2사이를 100개쪼개서 값을 np.array로 준다
print(x[:20])

#pyplot 데이터는 numpy,list모두가능
plt.plot(x, x, label='선형', color='g')  #green
plt.plot(x, x**2, label='2차', color='b') #blue
plt.plot(x, x**3, label='3차', color='r') #red

plt.title('제목')
plt.xlabel('x축')
plt.ylabel('y축')
plt.legend(loc="best")
plt.show()