import numpy as np
#numpy - 분석용라이브러리, 사실상 C언어
a = [1,2,3,4,5]
b = [2,4,6,8,10]
c = a+b
print(c)

#머신러닝이든 딥러닝 둘다 취급하는 데이터타입은 nd어레이 타입이다.
a = np.array(a) #타입을list->ndarray타입으로 (C언어의배열,속도빠름)
print(a, type(a))
b = np.array(b)