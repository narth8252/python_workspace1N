# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707\images
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-230p
#0708 pm 1시

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
path = "./머신러닝250707/images/mnist"
filenameList = os.listdir(path) #해당경로의 모든파일목록 전달
print(filenameList)
# ['cat.jpg', 'cat1.jpg', 'cat2.jpg', 'cat3.jpg', 'cat4.jpg', 'cat5.jpg', 'dog1.jpg', 'dog2.jpg', 'dog3.jpg', 'dog4.jpg', 'dog5.jpg']
#목록이 위와같이 나왔으니 하나씩 읽을수있다.
imgList = []
i=0
for filename in filenameList:
    filepath = path +"/" + filename
    temp = pilimg.open(filepath)
    #img크기축소
    img = temp.resize((80,80)) #tuple로 전달
    img = np.array(img) #이미지가져올때 3D아닌데 3D해놓고, 형식다르거나 하면 에러체크
    print(img.shape) #(80, 80, 3)으로 크기줄임
    imgList.append(img) #리스트에 추가하자
    i+=1
    if i > 10:
np.savez("data1.npz", data = imgList)
#통으로 저장. 확장자는변경불가

data1 = np.load("data1.npz")["data"]
plt.figure(figsize=(20, 5))
for i in range(1, len(data1)+1):
    plt.subplot(1, 11, i) #차트공간을 1 by 10개로 나누고 1부터 번호붙여
    plt.imshow(data1[i-1])
plt.show()

# #계속출력하고있어서 Ctrl+C로 




