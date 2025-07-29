# 와인 데이터셋처럼 수치형 데이터를 다룰 때는 CNN(합성곱 신경망)을 사용하지 않고 일반적인 딥러닝 모델, 즉 완전 연결(Dense) 층으로만 구성된 신경망으로도 충분합니다.
# 왜 와인 데이터셋에 CNN이 필요 없을까요?
# CNN은 주로 이미지나 영상, 음성과 같이 공간적 또는 시간적 패턴이 중요한 데이터에 특화된 딥러닝 모델입니다.
# 이미지 데이터: CNN은 필터(커널)를 사용하여 이미지의 픽셀 간의 **공간적 관계(예: 선, 모서리, 질감)**를 학습합니다. 얼굴의 윤곽선이나 고양이의 귀 모양 같은 특징은 픽셀들의 특정 배열에서 나타나죠.
# 와인 데이터셋: 와인 데이터셋은 13가지 화학 성분(예: 알코올 함량, 말산, 재)과 같은 수치형 특성들로 구성되어 있습니다. 이 특성들은 서로 특정한 공간적 또는 시간적 배열을 가지지 않습니다. 각 특성 값 자체가 독립적인 의미를 가지며, 이 값들의 조합으로 와인 종류가 결정됩니다.

import keras.utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras import models, layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import PIL.Image as pilimg
import imghdr
import pandas as pd
import pickle
import keras
import os
import shutil

original_dataset_dir = "../data/cats_and_dogs/train"

base_dir = "../data/cats_and_dogs_small"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

#ImageDataGenerator나 DataSet이나, 두 폴더보고 자동라벨링
train_cats_dir