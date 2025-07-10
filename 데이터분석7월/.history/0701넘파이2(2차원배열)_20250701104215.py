import numpy as np

#수학적으로 행렬이라고 부름(matrix)
m1 = np.array( [[1,2], [3,4]])
m2 = np.array( [[5,6], [7,8]])
"""
행렬의 더하기
1 2     5 6 = 1+5  2+6    6  8
2 3     7 8 = 2+7  3+8   10 12

"""
m3 = m1 + m2 
print(m3)

print(m1[0,0])
print(m1[0,1])
print(m1[1,0])
print(m1[0,0])