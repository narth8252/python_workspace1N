import pandas as pd
import numpy as np
import mglearn
import os
import matplotlib.pyplot as plt

#특성이 원래는 카테고리인데 숫자형태로 입력될때 처리방법
demo_df = pd.DataFrame({
    '숫자특성': [0,1,2,1], #범주형
    '범주형특성': ['양말', '여우', '양말', '상자']
})
#get_dummies : 숫자형의 범주데이터를 범주로 파악못함
df1 = pd.get_dummies(demo_df)
print(df1)

