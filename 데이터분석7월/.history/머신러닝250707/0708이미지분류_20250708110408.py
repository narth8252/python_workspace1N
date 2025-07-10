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

print("-------이미지 key값 추가됨-------")
print( data.images[:10]) #numpy 