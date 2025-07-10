#sklearn에서 load_digits() → 손으로쓴숫자 맞추기
#애초에 미국우편번호 나누기위해 개발돼 그때 수집된 데이터
#이미지 → 디지털화하는 과정에 흑백은2차원배열, 컬러는3차원배열임
#이미지가 10장있고 각이미지크기가 150 by 150
#10 150x150이 특성개수가 된다. 이미지를 읽어서 → numpy배열로 변경(오래걸리는데 파이썬PIL라이브러리로 제공)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1.데이터준비:현재데이터는 정리해서 준 데이터이지만, 실제데이터는 우리가 이 작업해야함
data = load_digits()    
X = data["data"]
y = data["target"]
print(X.shape)
print(X[:10])

print("-------이미지 key값 추가돼서 그림으로 그려줌-------")
print( data.images[:10]) #numpy 2D → 1D로 바꿔 data로 준거고 원래데이터
images = data.images
# #이미지1개출력
# plt.figure(figsize=(10, 4)) #차트의 크기
# plt.imshow(images[0], cmap="gray_r") #gray로 이미지출력
# plt.show()

#이미지여러개 동시출력하려면 화면분할(inch단위)
plt.figure(figsize=(10, 4)) #화면전체크기 지정후 작게 나눌시 subplot함수사용
# 2 by 5 로 쪼개면 10개의 화면만들어지고 각분할위치에 번호인덱스 붙음
# 1 2 3 4 5
# 6 7 8 9 10
for i in range(10):
    plt.subplot(2, 5, i+1) #내가 내보낼 위치지정
    plt.imshow(images[i], cmap="gray_r", interpolation='nearest') #없으면 옆색깔써 보간법
    plt.title(f"Label:{y[i]}")
    plt.axis('off') #축없애기

plt.tight_layout() #이쁘게 다시 정리해라
plt.subtitle("첫10개 Digits 이미지", y=1.05, font)
