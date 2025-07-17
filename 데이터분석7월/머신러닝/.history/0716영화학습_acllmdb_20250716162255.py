#250716 PM4시
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

#데이터로딩
# import os
# print(os.getcwd()) #현재 작업 폴더 확인 가능.
# 작업폴더→ C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\머신러닝
# DB폴더 → C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb
#상대경로
# 현재 작업폴더(머신러닝) → 상위폴더로 가서 ..\data\aclImdb\train 폴더로 가야함
reviews_train = load_files("../data/aclImdb/train")
#절대경로
# reviews_train = load_files(r"C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb\train")

print(reviews_train.keys)

# 텍스트와 레이블 분리
text_train = reviews_train.data         # list of raw byte strings (reviews)
y_train = reviews_train.target          # 0 or 1 labels

# 리스트를 판다스로 정리
df = pd.DataFrame({
    "text": [doc.decode("utf-8", errors="ignore") for doc in text_train],
    "target": y_train
})

# CSV 저장
df.to_csv("imdb.csv", encoding="utf-8-sig", index=False)

# 일부 출력
print(df.head())
#                                                 text  target
# 0  I caught this film at the Edinburgh Film Festi...       0
# 1  Based upon the recommendation of a friend, my ...       0
# 2  This is not an entirely bad movie. The plot (n...       0
# 3  I must confess to not having read the original...       1
# 4  as the title of this post says, the section ab...       2

#벡터화
#일종의 비지도학습, fit학습후 문장을 연산가능한 벡터