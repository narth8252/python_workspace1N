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
#첫번째특성에 대해서는 그냥 숫자로 받아들이고 두번째특성은 문자열의 경우만 범주로 받음
#첫번째특성을 문자열 또는 카테고리형으로 바꾸자
demo_df['숫자특성'] = demo_df['숫자특성'].astype(str)
df1 = pd.get_dummies(demo_df)
print(df1)

#OneHotEncoder 클래스 사용하자
from sklearn.preprocessing import OneHotEncoder
# sparse_output=False: false는 2가지리턴가능(희소행렬, numpy배열)
ohe = OneHotEncoder(sparse_output=False) 
on
