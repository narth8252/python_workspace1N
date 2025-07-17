import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from matplotlib import font_manager, rc
#font_manager - 폰트를 폰트객체화 
#rc - 폰트를 지정할 영역 (차트영역)

#절대경로지정
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
#상대경로지정 - 내컴폰트를 작업폴더에 복붙(malgun.ttf은기본깔림)
font_name = font_manager.FontProperties(fname="./fonts/malgun.ttf").get_name()
rc('font', family=font_name)
print(font_name)
plt.rcParams['f']