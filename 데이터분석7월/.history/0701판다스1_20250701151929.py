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
print(s[0])
