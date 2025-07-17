#250716 PM4시
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb_v1
from sklearn.datasets import load_files
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#데이터로딩
reviews_train = load_files("../data/aclImdb/train")
print(reviews_train.keys)

text_train = 