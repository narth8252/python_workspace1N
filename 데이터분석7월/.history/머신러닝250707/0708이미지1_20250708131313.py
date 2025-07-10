# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707\images
# 0708 pm 1시

import PIL.Image as pilimg
import numpy as np

img = pilimg.open("./머신러닝250707/images/1.jpg")
print(type(img)) 

#ndaray로 바뀐다
pix = np.array(img)
print(pix.shape)