import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from matplotlib import font_manager, rc
#font_manager - 폰트를 폰트객체화 
#rc - 폰트를 지정할 영역 (차트영역)

#절대경로지정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
#상대경로지정 - 내컴폰트를 작업폴더에 복붙(malgun.ttf은기본깔림)
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\fonts
#       스크립트위치	      폰트위치	                        상대경로지정
# ...데이터분석7월\머신러닝	...데이터분석7월\fonts\malgun.ttf	"../fonts/malgun.ttf"
# ...데이터분석7월	       ...데이터분석7월\fonts\malgun.ttf   "./fonts/malgun.ttf"
# font_name = font_manager.FontProperties(fname="./fonts/malgun.ttf").get_name()
font_name = font_manager.FontProperties(fname="../fonts/malgun.ttf").get_name()
# rc('font', family=font_name)
print(font_name)
plt.rcParams['font.family'] = font_name    #폰트설정
plt.rcParams['axes.unicode_minus'] = False #눈금에서 한글깨짐 문제발생