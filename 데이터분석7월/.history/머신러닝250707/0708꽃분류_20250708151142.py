# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-230p
#0708 pm 1시
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\data1\flowers
#daisy, dandelion, rose, sunfower, tulip 
#이 이미지들로 머신러닝하게 라벨링작업해라
#입력데이터는 무조건2D, 라벨링은 1D, 폴더별로 0,1,2,3,4

import PIL.Image as pilimg
import os #파이썬에서os명령어사용, 해당디렉토리 파일목록 통째로
import imghdr #이미지종류 알아낼때 사용
import numpy as np #이미지읽어서 넘파이배열로 바꾸기
import pandas as pd

#makeData 폴더명이랑 라벨주면 makdData("daisy", 1)
#해당폴더 데이터 쭉읽어서 numpy배열로 바꾸고 라벨링작업
def makeData(folder, label):

    data = [] #이미지피처저장 - 마지막에 둘다반환
    labels = [] #라벨저장
    path = "./data1/flowers"+folder
    #폴더이름 리스트를 가져와 정렬
    for filename in os.listdir(path): #폴더이쁘게 가져옴
        #이미지파일경로생성
        # image_path = os.path.join(folder_path, filename)
        try:
            print(path+"/"+filename)
            kind = imghdr.what(path +"/"+ filename) #파일종류확인위한 파이썬명령어
            #파일확장자잘라서 확인할까? 안됨(확장자는윈도우에만있음). 리눅스는 파일정보에 종류저장임
            if kind in ["gif", "png", "jpg", "jpeg"]: #이 이미지에 해당되면
                img = pilimg.open(path +"/"+filename)
                #img크기다르면 분석불가능, 동일한크기로 맞추기
                #img크기 너무크면 학습시 픽쳐개수많아서 힘드니 적당한크기로자르기
                resize_img = img.resize((80,80)) #크기를tup
                #이미지를 1D배열로 펼친후 2D배열로 간주해 추가
                #실제 CNN등에서는 2D그대로 쓰지만, 요구에맞춰 flatten
                image_data.append(img.flatten())
                labels.append(label)
        except Exception as e:
            print(f"{image_path} 파일처리중 오류발생: {e}")

    #리스트를 NumPy배열로 변환
    #이미지데이터는 2D, 라벨은 1D형태가 됨
    return np.array(image_data), np.array(labels)

#데이터셋 경로설정
data_path = r'C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\data1\flowers'




