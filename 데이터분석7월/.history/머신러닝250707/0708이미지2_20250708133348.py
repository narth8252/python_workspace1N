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
# img.save("./머신러닝250707/images/1.bmp")
# 현재 (340, 514, 3) → 이미지크기 340 x 514 x 3차원 
# 80 x 80 x 3차원으로 줄여서 numpy계열로 저장해보자
path = "./머신러닝250707/images/animal"
filenameList = os.listdir(path) #해당경로의 모든파일목록 전달
print(filenameList)
# ['cat.jpg', 'cat1.jpg', 'cat2.jpg', 'cat3.jpg', 'cat4.jpg', 'cat5.jpg', 'dog1.jpg', 'dog2.jpg', 'dog3.jpg', 'dog4.jpg', 'dog5.jpg']
#목록이 위와같이 나왔으니 하나씩 읽을수있다.
for filename in filenameList:
    filepath = path +"/" + filename
    temp = pilimg.open(filepath)
    #img크기축소
    img = temp.resize((80,80)) #tuple로 전달
    img = np.array(img) #이미지가져올때 3D아닌데 3D해놓고, 형식다르거나 하면 에러체크
    print(img.shape) #(80, 80, 3)으로 크기줄

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




