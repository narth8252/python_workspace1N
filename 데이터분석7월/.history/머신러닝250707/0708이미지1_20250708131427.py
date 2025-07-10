# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707\images
# 0708 pm 1시

import PIL.Image as pilimg
import numpy as np

img = pilimg.open("./머신러닝250707/images/1.jpg")
print(type(img)) 

pix = np.array(img) #ndaray로 바뀐다
print(pix.shape) #컬러이미지라 3D 나온다

for i in range(pix.shape[0]):
    for j in range(pix.shape[1]):
        