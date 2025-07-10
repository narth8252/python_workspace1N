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
