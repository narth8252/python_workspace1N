import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import font_manager

# 1. 한글 폰트 설정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()

# 2. Seaborn 스타일 세팅
sns.set_style('whitegrid')

# 3. 폰트 설정 + 마이너스 깨짐 방지
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 4. 데이터 생성
x = np.random.normal(loc=70, scale=20, size=1000)

# 5. 히스토그램 시각화
sns.displot(x, bins=20, kde=True, rug=False)

plt.title('히스토그램')
plt.xlabel('x축')
plt.ylabel('y축')
plt.show()
