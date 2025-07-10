import numpy as np
# numpy - 분석용라이브러리, 사실상 C언어
a = [1,2,3,4,5]
b = [2,4,6,8,10]
c = a+b
print(c)

# 머신러닝이든 딥러닝 둘다 취급하는 데이터타입은 nd어레이 타입이다.
a1 = np.array(a) # 타입을 list->ndarray타입으로 (C언어의 배열, 속도빠름)
print(a1, type(a1))
b1 = np.array(b)
c1 = a1 + b1    # 수학의 벡터연산을 수행
                # 스칼라연산 - 요소 하나씩 더하기
                # 벡터연산 - 벡터 통채로 (배열통으로 연산수행, forX)
print(c1)

print("평균", np)