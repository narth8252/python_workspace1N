# C:\Users\Admin\Documents\넘파이,판다스\파이썬데이터분석(배포X)\9차시_백현숙

import numpy as np
import pandas as pd

a = [1,2,3,4,5]
s = pd.Series(a)
print(type(s))
print(s)
b={'a':1, 'b':2, 'c':3, 'd':4, 'e':5 }
s2 = pd.Series(b) #dict을 시리즈타입으로

print(s[0])
print(s[1])
print(s[2])
print("-----------------------------")
print(s[:3])
print(s[s>=3]) #조건식가능
print("-----------------------------")
#넘파이랑 시리즈랑 사용방법 유사함.

#3보다 크거나 같고 4보다 작거나 같다.
print(s[np.logical_and(s>=3, s<=4)])
print(s[[0,3,2]])
print("-----------------------------")
print(s2[:3])
print(s2["a"])
print(s2["a":"c"])
# print(s2["d","a","c"])
# print(s2[4,2,1])
print(s2.loc[["d","a","c"]])  # 여러 라벨 인덱싱은 loc 사용
print(s2.iloc[[4,2,1]])       # 여러 정수 인덱싱은 iloc 사용
"""
import pandas as pd

a = [1,2,3,4,5]
s = pd.Series(a)    #  list => Series 타입으로 만들기
print(type(s))
print(s)

b = {"a":1, "b":2, "c":3, "d":4, "e":5}
s2 = pd.Series(b)   # dict => Series
print(s2)    

print(s[0])
print(s[1])
print(s[2])
print(s[:3])
print(s[s>=3])  # 조건식 가능함

import numpy as np
# 3보다 크거나 같고 4보다 작거나 같다.
print(s[np.logical_and(s>=3, s<=4)])
print(s[[0,3,2]])

print("="*20)
print(s2[:3])
print(s2["a"])
print(s2["a":"c"])
print(s2[["d", "a", "c"]])
print(s2[[4,2,1]])
"""