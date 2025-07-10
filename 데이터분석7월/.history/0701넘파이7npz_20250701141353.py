import numpy as np
data1=np.arange(1,11)
data2=np.random.rand(10)
print(data1)
print(data2)
**확장자가.npz**케클에서 제공하는 이미지가 이 파일임.
Np.savez( ＇data.npz＇, key1=data1, key2=data2)
#미국에서 우편번호가 너무많아서 손글씨를 파악해서 자동분류하려고 이미지분류 npz로 만들어서 딥러닝에서 읽어낼수있으려면 그래서 .npz파일이다.
#함수명이 savez 이다 파일 확장자는 npz이고, key=값 형태로 저장한다 
outfile = np.load('data.npz')
print(outfile.files) 
data1 = outfile['key1']
data2 = outfile['key2']
print(data1, data2)
