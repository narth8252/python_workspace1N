# 250716 PM1시 
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
sns.set_style('whitegrid') #{darkgrid, whitegrid, dark, white, ticks}

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