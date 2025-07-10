# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-230p
#0708 pm 1시
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\data1\flowers
#daisy, dandelion, rose, sunfower, tulip 
#이 이미지들로 머신러닝하게 라벨링작업해라
#입력데이터는 무조건2D, 라벨링은 1D, 폴더별로 0,1,2,3,4

import PIL.Image as pilimg
import os #파이썬에서os명령어사용, 해당디렉토리 파일목록 통째로
import cv2
import matplotlib.pyplot as plt #모든그래픽출력은 pyplot으로 해야함
import numpy as np #이미지읽어서 넘파이배열로 바꾸기

def label_from_folder(main_folder_path):
    """
    폴더에 담긴 이미지들을 라벨링하여 머신러닝용 데이터셋을 생성합니다.
    Args:
        main_folder_path (str): 'daisy', 'dandelion' 등 클래스 별 폴더가 들어있는 메인 폴더 경로.
    Returns:
        (numpy.ndarray, numpy.ndarray): 이미지 데이터 (2D 배열), 라벨 (1D 배열).
    """
path = "./data1/flowers/daisy"

imgList = []
labels = []

#폴더이름 리스트를 가져와 정렬
folder_name = sorted(os.listdir(main_folder_path))
label_map = {folder_name: i for i, folder_name in enumerate(folder_name)}
print(f"라벨 맵: {label_map}")

for folder_name, label in label_map.items():
    fol = path +"/" + filename
    temp = pilimg.open(filepath)
    #img크기축소
    img = temp.resize((80,80)) #tuple로 전달
    img = np.array(img) #이미지가져올때 3D아닌데 3D해놓고, 형식다르거나 하면 에러체크
    print(img.shape) #(80, 80, 3)으로 크기줄임
    imgList.append(img) #리스트에 추가하자
    i+=1
    if i >= 10:
        break
np.savez("flower1.npz", data = imgList)
#통으로 저장. 확장자는변경불가

data1 = np.load("flower1.npz")["data"]
plt.figure(figsize=(20, 5))
for i in range(1, len(data1)+1):
    plt.subplot(1, 11, i) #차트공간을 1 by 10개로 나누고 1부터 번호붙여
    plt.imshow(data1[i-1])
plt.show()

#계속출력하고있어서 Ctrl+C중단




