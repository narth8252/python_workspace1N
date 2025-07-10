import numpy as np

data = np.random.rand(5)
print(data)
#저장
np.save(‘datafile.npy’, data) #데이터1개만 저장가능
#파일확장자는 npy이다, 생략하면 무조건 npy를 부여한다. 다른건안됨.
np.array 1개만 저장됨.  
data=[]
print(data)
data = np.load(‘datafile.npy’)
print(data)
