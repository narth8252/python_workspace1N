#쌤PPT-24p.
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)\11차시_백현숙\[평생]원고_v1.0_11회차_데이터셋_백현숙_0915_1차.pptx
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\data
# auto-mpg.csv
#파일명 : exam11_.py
#데이터표준화

#구간나누기 코딩
#bins = 나눠야할 구간의 개수
#구간을 나눠서 각 구간별 데이터개수와 구간에 대한 정보를 바노한
import numpy as np

count, bin_dividers = np.histogram(data['power'], bins=4)
print("각 구간별 데이터 개수 : ", count)
print("구간정보 : ", bin_dividers)