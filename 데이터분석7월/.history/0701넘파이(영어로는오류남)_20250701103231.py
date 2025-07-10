import numpy as np
from scipy import stats
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


print("평균:", np.mean(c1))
print("중간값(중위수):", np.median(c1))
print("최댓값:", np.max(c1))
print("최솟값:", np.min(c1))
print("표준편차:", np.std(c1))
print("분산:", np.var(c1))
print("최빈값:", stats.mode(c1, keepdims=True).mode[0])

"""
[1, 2, 3, 4, 5, 2, 4, 6, 8, 10]
→ 리스트 a와 b를 더한 결과(리스트 연결)

[1 2 3 4 5] <class 'numpy.ndarray'>
→ a를 numpy 배열로 변환한 결과와 타입

[ 3 6 9 12 15]
→ numpy 배열 a1과 b1의 벡터 합

평균: 9.0
→ c1의 평균

중간값(중위수): 9.0
→ c1의 중앙값

최댓값: 15
→ c1의 최댓값

최솟값: 3
→ c1의 최솟값

표준편차: 4.242640687119285
→ c1의 표준편차

분산: 18.0
→ c1의 분산

최빈값: 3
→ c1에서 가장 많이 나온 값(여기서는 모든 값이 1번씩만 나와서 첫 번째 값이 반환됨)
"""