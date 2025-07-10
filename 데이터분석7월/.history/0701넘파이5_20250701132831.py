# 0701 쌤PPT에 없음
#argmax, argmin
import numpy as np

#랜덤값은 컴퓨터의 내부의
#보고서 쓸때 값이 계속 바뀌면 문제됨
#고정시키고싶을때 seed값
a = np.random.rand(5)
print(a)
print(np.max(a), np.argmax(a)) #argmax는 큰값이 있는 위치값반환
