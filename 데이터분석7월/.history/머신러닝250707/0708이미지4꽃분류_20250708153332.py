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
    # 정확한 상대 경로 지정 (슬래시 통일)
    #base_path = "C:/Users/Admin/Documents/GitHub/python_workspace1N/데이터분석250701/data1/flowers"
    # folder는 예: "/daisy" 이런 형식이라면 앞에 슬래시 제거 필요
    #folder = folder.lstrip("/")

    data = [] #이미지피처저장 - 마지막에 둘다반환
    labels = [] #라벨저장
    path = ".data1/flowers"+folder
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
                resize_img = img.resize((80,80)) #크기를tuple로 전달하면 앞의 이미지크기만 변경
                pixel = np.array(resize_img) #img → ndarray로 변경
                if pixel.shape==(80,80,3): #이미지크기 같은것만 취급
                    data.append(pixel)
                    labels.append(label)
        except:
            print(filename +f"파일처리중 오류발생") #어떤파일이 에러인지 직접찾아삭제위함

    #파일로 저장해서 담에 사용또하자
    np.savez("{}.npz".format(folder), data=data, targets=labels)

#1.파일로 저장하기
def filesave():
    makeData("daisy", "0")
    makeData("dandelion", "1")
    makeData("rose", "2")
    makeData("sunflower", "3")
    makeData("tulip", "4")

#2.
def loadData():
    daisy=np.load("daisy.npz")
    dandelion=np.load("dandelion.npz")
    rose=np.load("rose.npz")
    sunflower=np.load("sunflower.npz")
    tulip=np.load("tulip.npz")

    data = np.concatenate((daisy["data"], dandelion["data"], rose["data"], \
                           sunflower["data"], tulip["data"]))
    target = np.concatenate((daisy["data"], dandelion["data"], rose["data"], \
                           sunflower["data"], tulip["data"]))
    print(data.shape)
    print(target.shape)
    return data, target #지역변수라서 함수내에서만 존재리턴해주자

data, target = loadData()
data = data.reshape(data.shape[0], 80*80*3) 
#4D머신딥러닝 → 딥러닝(CNN,합성곱신경망) → CNN 은 차원을 그대로 받아들임
print(data.shape)





