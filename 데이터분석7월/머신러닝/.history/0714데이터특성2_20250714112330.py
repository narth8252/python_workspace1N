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
# demo_df['숫자특성'] = demo_df['숫자특성'].astype(str)
# df1 = pd.get_dummies(demo_df)
# print(df1)

print("-------------------------------------------")
#OneHotEncoder 클래스 사용하자
from sklearn.preprocessing import OneHotEncoder
# sparse_output=False: false는 2가지리턴가능(희소행렬, numpy배열)
ohe = OneHotEncoder(sparse_output=False) 
# OneHotEncoder 반환값을 별도로 받아야하는데 리턴값이 numpy배열형태임
n = ohe.fit_transform(demo_df) #fit → transform
print(ohe.fit_transform()) #필드이름
print(n) #입력값


import pandas as pd
import numpy as np
import mglearn
import os
import matplotlib.pyplot as plt

# 숫자처럼 생겼지만 사실상 범주형 데이터
demo_df = pd.DataFrame({
    '숫자특성': [0, 1, 2, 1],  # ← 범주형 숫자
    '범주형특성': ['양말', '여우', '양말', '상자']
})

# get_dummies: 문자열 특성만 자동 인식
df1 = pd.get_dummies(demo_df)
print("[get_dummies 결과]")
print(df1)

print("-------------------------------------------")

# 숫자형 범주를 정확히 처리하려면 OneHotEncoder가 더 유리
from sklearn.preprocessing import OneHotEncoder

# sparse_output=False: 결과를 배열로 받기
ohe = OneHotEncoder(sparse_output=False)

# fit_transform → 입력필요함
n = ohe.fit_transform(demo_df)

# 열 이름 확인 (sklearn >=1.0 에서 가능)
print("[인코딩된 컬럼 이름]")
print(ohe.get_feature_names_out())

# 결과 출력
print("[OneHotEncoder 변환 결과]")
print(n)
