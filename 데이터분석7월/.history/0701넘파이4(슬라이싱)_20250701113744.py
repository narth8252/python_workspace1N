import numpy as np

x = np.arange(20)
print(x)
print(x[:10]) #0~9번방까지
print(x[10:]) #10번방~끝까지
print(x[::-1]) #0~끝번방까지 역순으로
print(x[10:2:-1])
print(x[10:0:-2])
print(x[10:2:-1]) 