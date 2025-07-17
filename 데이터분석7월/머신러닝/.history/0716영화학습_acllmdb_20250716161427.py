#250716 PM4시
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb
from sklearn.datasets import load_files
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#데이터로딩
# import os; 
# print(os.getcwd()) #현재 작업 폴더 확인 가능.
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\머신러닝
# reviews_train = load_files("./data/aclImdb/train")
#절대경로
reviews_train = load_files(r"C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb\train")

print(reviews_train.keys)
# #로딩오래걸리니 여기서 저장해놓고 쓰자
text_train = reviews_train.data
y_train = reviews_train.target

df = pd.DataFrame(text_train, columns=text_train.feature_names)
df["target"] = y_train
df.to_csv
print(text_train.head())