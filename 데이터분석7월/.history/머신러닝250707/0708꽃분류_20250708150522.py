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
# path = "./data1/flowers/daisy"
    data = [] #이미지피처저장 - 마지막에 둘다반환
    labels = []

#폴더이름 리스트를 가져와 정렬
folder_name = sorted(os.listdir(main_folder_path))
label_map = {folder_name: i for i, folder_name in enumerate(folder_name)}
print(f"라벨 맵: {label_map}")

for folder_name, label in label_map.items():
    folder_path = os.path.join(main_folder_path, folder_name)
    
    #폴더가 아니면 건너뛰기
    if not os.path.isdir(folder_path):
        continue
    for filename in os.listdir(folder_path):
        #이미지파일경로생성
        image_path = os.path.join(folder_path, filename)

        try:
            #이미지를 흑백으로 불러오기
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
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




