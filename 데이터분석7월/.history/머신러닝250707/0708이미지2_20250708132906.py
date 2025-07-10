# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707\images
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-230p

# # 0708 pm 1시

import PIL.Image as pilimg
import os #파이썬에서 os명령어 사용하게 해준다
            #디렉토리검색해서 해당디렉토리 파일목록을 통째로
import matplotlib.pyplot as plt #모든그래픽출력은 pyplot으로 해야함
import numpy as np #이미지읽어서 넘파이배열로 바꾸기

#특정폴더의 이미지를 읽어서 전부 numpy배열로 바꾸고 다 더해서 npz파일로 저장하기
#이미지(4D,3D이미지가 여러장이라)를  2D ndarray로 바꾸는 방법 배우기
#여태까지 사이킷런, 케라스 등이 대신 해줌. 실제데이터는 내가해야함.

#형식바꿔서 저장해보자
img.save("./머신러닝250707/images/1.bmp")
# 현재 (340, 514, 3) → 이미지크기 340 x 514 x 3차원 
# 80 x 80 x 3차원으로 줄여서 numpy계열로 저장해보자
path = "./머신러닝250707/images/animal"
filenameList = os.listdir(path)



# img = pilimg.open("./머신러닝250707/images/1.jpg")
# print(type(img)) 

# pix = np.array(img) #ndaray로 바뀐다
# print(pix.shape) #컬러이미지라 3D 나온다

# print(pix)
# # for i in range(pix.shape[0]):
# #     for j in range(pix.shape[1]):
# #         for k in range(pix.shape[2]):
# #             print("{0:3}".format(pix[i][i][k], end=' '))
# #     print() 
# #계속출력하고있어서 Ctrl+C로 




