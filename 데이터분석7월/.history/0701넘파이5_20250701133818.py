# 0701 쌤PPT에 없음
#argmax, argmin
import numpy as np

#랜덤값은 컴퓨터의 내부의 시계이용해 추출하는데
#보고서 쓸때 값이 계속 바뀌면 문제됨
#고정시키고싶을때 seed값
np.random.seed(1234) #()안은 0~아무값이나 주면된다. 예제에도 의미X
a = np.random.rand(5)
print(a)
print(np.max(a), np.argmax(a)) #argmax는 큰값이 있는 위치값반환

a = np.random.rand(5)
print(a)
print(np.max(a), np.argmax(a)) #argmax는 큰값이 있는 위치값반환

#문제1.가우스분포에 따른 랜덤값을  5개씩 10개생성해서 
#각행마다 젤 큰값과 큰값위치 찾아출력하기
a = np.random.rand(50) #50개 일단 만들어놓고
a = a.reshape(5,10)
for i in range(0, 10):
    print( np.aremax)
