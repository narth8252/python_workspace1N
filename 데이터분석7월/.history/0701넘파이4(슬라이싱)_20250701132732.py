import numpy as np

x = np.arange(20)
print(x) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
print(x[:10]) #0~9번방까지 [0 1 2 3 4 5 6 7 8 9]
print(x[10:]) #10번방~끝까지 [10 11 12 13 14 15 16 17 18 19]
print(x[::-1]) #0~끝번방까지 역순으로
print(x[10:2:-1]) #[10  9  8  7  6  5  4  3]
print(x[10:0:-2]) #[10  8  6  4  2]
print(x[1:3]) #[1 2]
print(x[2:7]) #[2 3 4 5 6]
print("---------------------------------")
#조건식
print(x >= 10) #리스트가아님.독특한구조임.
# [False False False False False False False False False False  True  True
#   True  True  True  True  True  True  True  True]

#파이썬의 리스트는 조건식적용안되고 에러
# TypeError: '>=' not supported between instances of 'list' and 'int'
# a = [1,2,3,4,5]
# print(a>=3)

#R언어에서 가져온방식으로 뽑아내기 가능
#이거 넘파이만되지, 파이썬은 안되니 헷갈려하는 경우 많음
print( x[ [1,3,5,7,9]]) #[1 3 5 7 9] 
print( x[ x>=10])  #[10 11 12 13 14 15 16 17 18 19]
print("---------------------------------")
#x값이 짝수의 경우만
print( x[x%2==0])  #[ 0  2  4  6  8 10 12 14 16 18]
#넘파이 세상편함. 컴프리헨션안써도 돼서. 그래서 분석에 유용
print("---------------------------------")
#3의배수이면서 5의배수만 추출
# print( x[x%3==0 and x%5==0]) #이건안받아줌:and연산은 넘파이가아니라 파이썬임.
# The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
# 조건식 이렇게 해야하는거 중요함. 그래서 아래처럼 써야함. np.logical_
print(x[np.logical_and(x%3==0, x%5==0)])
#데이터분석가가 주로 써야할 함수라 중요
x = np.array([1,2,3,4,5])
print(x[[ True, True, True, False, False ]])

"""
[ 0 15]
[1 2 3]
"""
print("---------------------------------")
#2차원 배열에서 슬라이싱은 [행, 열] 형태로 사용
k = np.array([ [1,2,3,4,5],
[6,7,8,9,10],
[11,12,13,14,15],
[16,17,18,19,20]])

k2 = np.arange(1, 21)
k2 = k2.reshape(4, 5)
print(k)
print(k2)
print("---------------------------------")
# k[:] 또는 k[:, :]는 2차원 배열의 모든 행과 모든 열을 의미합니다.
print(k[:])      # 전체 행렬(모든 행, 모든 열)
print(k[:, :])   # 위와 동일하게 전체 행렬
print(k[:1]) #1행만
print(k[:2]) #첫2행(0~1행) 전체
print(k[:3]) #3행까지
print("---------------------------------")
print(k[:, :2])  #첫2열(0~1열)만 출력
print("---------------------------------")
print(k[::2]) 
# [[ 1  2  3  4  5]
#  [11 12 13 14 15]]
print("---------------------------------")
print(k[2:4, 3:5]) 
# [[14 15]
#  [19 20]]
print("---------------------------------")
#numpy배열하기
a = np.array([1,2,3,4,5])
b = np.arange(6, 11)
c = np.concatenate((a, b, np.arange(11,21)) ) #tuple로 받아간다.
print(c) #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
print("---------------------------------")
c1 = np.array_split(c, 3) #ndarray를 쪼개서 새로운배열로 만든다.
print(c1) #[array([1, 2, 3, 4, 5, 6, 7]), array([ 8,  9, 10, 11, 12, 13, 14]), array([15, 16, 17, 18, 19, 20])]

print("--------#검색-------------------------")
#검색은 where사용, 잘안씀
print(np.where(a%2==0)) #(array([1, 3], dtype=int64),)

print("--------#정렬-------------------------")
#정렬
c1 = np.random.randint(1, 100, 10) #1~100까지 랜덤10개 추출
print(c1)
c2 = np.sort(c1) #정렬된 새 배열을 반환, 원본은 그대로
print(c2)
c1.sort()        # 자기 자신을 정렬 (원본 배열이 바뀜)
print(c1)

np.sort(c1) : 정렬된 새 배열을 반환, 원본은 그대로
c1.sort() : 원본 배열 자체를 정렬(자기 자신이 바뀜)