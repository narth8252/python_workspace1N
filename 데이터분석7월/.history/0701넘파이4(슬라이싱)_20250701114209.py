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
print( x[ [1,3,5,7,9]]) #[1 3 5 7 9] 