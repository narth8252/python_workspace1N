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
A B C B  이렇게 원핫인코딩이다. 이게 행렬구조다. 단어가 몇만개면?
1 0 0 0
0 1 0 0 
0 0 1 0 
0 0 0 1

머신러닝에서 수치화와 타입 전환
ㆍ머신러닝은 모든 데이터가 숫자(수치형)여야 학습이 가능합니다.
ㆍㆍ만약 데이터에 **문자(예: A, B, C, D)**가 있다면,
→ 카테고리 타입(범주형)으로 변환해야 합니다.
카테고리화 할 수 없는 문자열(예: 이름, 고유값 등)은
→ 삭제하는 것이 일반적입니다.
ㆍ예시: 연비 등급(A, B, C, D)이 있을 때
A, B, C, D를 각각 1, 2, 3, 4로 바꿀 수 있지만,
이 숫자에 순서나 크기 의미가 없으므로
→ 동등한 값으로 처리해야 합니다.
ㆍ원-핫 인코딩(One-hot encoding)
각 등급을 새로운 컬럼으로 만들어서
해당 등급이면 1, 아니면 0으로 표시합니다.
  등급  연비A 연비B 연비C 연비D
    A	 1	  0    0	 0
    B	 0	  1	   0  	 0
    C	 0    0	   1	 0
    B	 0	  1	   0	 0
이렇게 하면 컴퓨터가 각 등급을 동등하게 인식합니다.
이 과정을 원-핫 인코딩이라고 합니다.

ㆍ머신러닝에서는 모든 데이터를 숫자로 변환해야 한다.
ㆍ범주형 데이터는 원-핫 인코딩 등으로 변환한다.
ㆍ의미 없는 문자열은 삭제한다.

"""
 #타입이 맞지 않을 경우 전환을 해서 사용해야 한다 ⭐
#현재 사용하는 파이썬 버전은 문자열 데이터라도 수치 형태면 자동으로 수치자료로 처리한다 
#파이썬 버전에 따라 다르게 동작할 수 도 있다 
data.columns=['mpg', 'cyl', 'disp', 'power', 'weight', 'acce', 'model']
print(data.dtypes)
print(data.head())
print( data['disp'].unique())

#정규식 sup함수
#잘못된 데이터를 NaN으로 먼저 바꾼다
# 데이터를 범주형으로 바꿀때 astype(‘category’)를 사용합니다. 
# 문자열의 형태로 입력된 데이터는 분석시에 꼭 범주형으로 바꾸어서 처리해야 합니다

#disp컬럼의 잘못된값 '?'을 결측치(NaN)로 바꾸고, 결측치있는행 삭제한뒤,
#disp를 float타입으로, model을 범주형(category)으로 변환 과정
#1.잘못된 데이터를 NaN으로 먼저 바꾼다
#'disp'컬럼에 값이 '?'로 잘못 입력된부분을 결측치(NaN)로 바꿉니다.
data['disp'].replace('?', np.nan, inplace=True)
print(data.head())

# 2.결측치가 있는 행 삭제
# disp 컬럼에 NaN이 있는 행(레코드)을 데이터프레임에서 삭제
data.dropna(subset=['disp'], axis=0, inplace=True)
print(data.head())

# 3.데이터타입 확인
print(data.dtypes) #→ 각컬럼의 데이터타입을 출력해서 현재 상태를 확인합니다.
# 4.
# disp 컬럼의 데이터 타입을 float로 변환한다
data['disp'] = data['disp'].astype('float')
print(data.dtypes)

# model 컬럼을 범주형(category)으로 변환한다
data['model'] = data['model'].astype('category')
print(data.dtypes)

