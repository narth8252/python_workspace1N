import numpy as np
import pandas as pd

a = [1,2,3,4,5]
s = pd.Series(a)
print(type(s))
print(s)
b={'a':1, 'b':2, 'c':3, 'd':4, 'e':5 }
series2 = pd.Series(b) #dict을 시리즈타입으로

print(s[0])
print(s[1])
print(s[2])
print("-----------------------------")
print(s[:3])
print(s[s>=3]) #조건식가능
print("-----------------------------")
#넘파이랑 시리즈랑 사용방법 유사함.
import numpy as np
#3보다 크거나 같고 4보다 작거나 같다.
print( s[np.logical_and(s>=3, s<=4)])
print(s[[0,3,2]])
print("-----------------------------")
print(s2[:3])
print(s2["a"])
print(s2["a":"c"])
print(s2[:3])