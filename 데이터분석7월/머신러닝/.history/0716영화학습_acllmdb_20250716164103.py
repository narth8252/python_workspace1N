#250716 PM4시
# C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# import os
# print(os.getcwd()) #현재 작업 폴더 확인 가능.
# 작업폴더→ C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\머신러닝
# DB폴더 → C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb
#상대경로
# 현재 작업폴더(머신러닝) → 상위폴더로 가서 ..\data\aclImdb\train 폴더로 가야함
#절대경로
# reviews_train = load_files(r"C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data\aclImdb\train")

# 1. 데이터 로딩
reviews_train = load_files("../data/aclImdb/train")
# 텍스트와 레이블 분리
text_train = reviews_train.data         # list of raw byte strings (reviews)
y_train = reviews_train.target          # 0 or 1 labels

# 2. bytes → 문자열 변환 + <br/> 제거
text_train_clean = [
    doc.decode("utf-8", errors="ignore").replace("<br />", " ") 
    for doc in text_train
]

# 3. DataFrame 저장(선택사항) 리스트를 판다스로 정리 -불러오는데 오래걸리니까
df = pd.DataFrame({
    "text": [doc.decode("utf-8", errors="ignore") for doc in text_train],
    "target": y_train
})

# CSV 저장, 일부출력
df.to_csv("imdb.csv", encoding="utf-8-sig", index=False)
print(df.head())
#                                                 text  target
# 0  I caught this film at the Edinburgh Film Festi...       0
# 1  Based upon the recommendation of a friend, my ...       0
# 2  This is not an entirely bad movie. The plot (n...       0
# 3  I must confess to not having read the original...       1
# 4  as the title of this post says, the section ab...       2


#문자열앞에 b → binary 약자 없애자, br태그 등 불필요한것 없애기
# text_train = [i.replace(b"<br/>", "b") for i in text_train]

# 4. 벡터화
#일종의 비지도학습, fit학습후 문장을 연산가능한 벡터로변환
vect = CountVectorizer().fit(text_train_clean)
X_train = vect.transform(text_train_clean)

#피처이름확인 - 컬럼명 생성
feature_names = vect.get_feature_names_out()
print("특성개수", len(feature_names))
print("특성20개만 확인", feature_names[:20])


# 모델 학습
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)  # 경고 방지를 위한 반복 횟수 증가
model.fit(X_train, y_train)
# 학습 정확도 (※ 테스트셋 아님)
print("학습 정확도:", model.score(X_train, y_train))
# print(model.score(X_train, y_train))

reviews_test = load_files("../data/aclImdb/test")
text_test_clean = [doc.decode("utf-8", errors="ignore").replace("<br />", " ") for doc in reviews_test.data]
X_test = vect.transform(text_test_clean)
y_test = reviews_test.target

print("테스트 정확도:", model.score(X_test, y_test))
