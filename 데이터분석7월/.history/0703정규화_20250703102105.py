"""
머신러닝의 최초
y = ax + b 
통계학자들이 손풀이로 이렇게 해봄
자동으로 a와 b를 찾아내는 과정
a, b    2 1
x = [1,2,3,4,5....]
y = [.............] 기대값
real_y= [.........] 오차의 제곱의합: 454678
a,  b에 다른값 넣어봐.. 3 1

선형회귀분석이 알고리즘의 시작.
y = w1x1 + w2x2 + w3x3 + w4x4 .... + b
y에 영향을 미치는 x1, x2, ............정해져있음
          기울기 w1, w2, ...........가중치를 찾아내는 과정

모든 데이터(필드, 특성, 픽처)들의 단위가 비슷해야한다.
모든 머신러닝이 같아야하는건 아니지만 몇가지 반드시 맞춰야하는 것이 있다.
서포트벡터머신(정확도 높음) 머신러닝, 딥러닝 : 반드시 단위비슷해야한다.
정규화 : 단위를 맞추는 과정
"""
#파일명 : exam11_4.py
#데이터표준화
import pandas as pd 
data = pd.read_csv('./data/auto-mpg.csv')
print(data.info())
print(data.head())
#컬럼명 변경하기 
data.columns=['mpg', 'cyl', 'disp', 'power', 'weight', 'acce', 'model']
print(data.head())

#정규화 
#(정규화하고자 하는 값 - 데이터 값들 중 최소값) / (데이터 값들 중 최대값 - 데이터 값들 중 최소값)
data['mpg2'] = (data['mpg'] - data['mpg'].min())/(data['mpg'].max()-data['mpg'].min())
print(data)
#단위환산 - 한국 단위로 환산하기 
mpg_unit = 1.60934 / 3.78541
data['kpl'] = (data['mpg'] * mpg_unit).round(2)
print( data.head() )
