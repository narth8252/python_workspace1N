#쌤PPT-21p.
# c:\Users\Admin\Documents\250701넘파이,판다스\파이썬데이터분석(배포X)\11차시_백현숙\[평생]원고_v1.0_11회차_데이터셋_백현숙_0915_1차.pptx
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701딥러닝\data
# auto-mpg.csv
#파일명 : exam11_5.py
#데이터표준화
import pandas as pd
import numpy as np
data = pd.read_csv('./data/auto-mpg.csv')
print(data.info())
print(data.head())
"""
머신러닝 - 수치화가 가능해야 머신러닝가능
연비 -> A, B, C, D -> 문자열 -> 카네고리타입으로 A, B,C,D 이 타입을 제외한
나머지 타입은 프로그램으로 넣을수없게 차단해야한다.
카테고리화 할수없는 문자열(이름..)필드는 삭제해야함
A - 1
B - 2
C - 3
D - 4
이걸 동등가치로 컴퓨터가 받아들여야되는데 숫자 하나하나가 중요한숫자니까
그래서 아래처럼 새로운 필드로 컬럼으로 만들어버림.

연비 하나 올수있었다.A B C B 
 필드를 4개로 바꾼다. - 연비1, 연비2, 연비3, 연비4
A B C B  이렇게 원핫인코딩이다. 이게 행렬구조다. 단어가 몇만개
1 0 0 0
0 1 0 0 
0 0 1 0 
0 0 0 1

"""
 #타입이 맞지 않을 경우 전환을 해서 사용해야 한다 ⭐
#현재 사용하는 파이썬 버전은 문자열 데이터라도 수치 형태면 자동으로 수치자료로 처리한다 
#파이썬 버전에 따라 다르게 동작할 수 도 있다 
data.columns=['mpg', 'cyl', 'disp', 'power', 'weight', 'acce', 'model']
print(data.dtypes)
print(data.head())
print( data['disp'].unique())
