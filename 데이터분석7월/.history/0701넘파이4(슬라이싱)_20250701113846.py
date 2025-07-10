import numpy as np

x = np.arange(20)
print(x) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
print(x[:10]) #0~9번방까지 [0 1 2 3 4 5 6 7 8 9]
print(x[10:]) #10번방~끝까지 [10 11 12 13 14 15 16 17 18 19]
print(x[::-1]) #0~끝번방까지 역순으로
print(x[10:2:-1])
print(x[10:0:-2])
print(x[1:3])
print(x[2:7])

print(x)