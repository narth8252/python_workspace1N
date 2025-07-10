import numpy as np
#numpy - 분석용라이브러리, 사실상 C언어
a = [1,2,3,4,5]
b = [2,4,6,8,10]
c = a+b
print(c)

a = np.array(a) #타입을list->ndarray타입으로 (C언어의배열,속도빠름)