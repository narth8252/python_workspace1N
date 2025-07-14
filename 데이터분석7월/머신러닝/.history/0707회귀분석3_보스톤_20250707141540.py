# 250707 pm2:10 250701딥러닝_백현숙.PPT-285p
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-p
# 저장폴더 C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707

http://lib.stat.cmu.edu/datasets/boston
#다중회귀분석: 공분산을 따져보고 특성제거해야함
#R은 기본적으로 제거하지만, 파이썬은 내가 해야함.
#powershell은 파일명에()를 못읽음
import pandas as pd #다양한유형의 데이터있을때 처리방법
import numpy as np
#분기문자가 공백이 아니고
df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
print(df.head(10))


# from sklearn.linear_model import LinearRegression
# model = LinearRegression() #하이퍼파라미터없음(과대,과소던 할수있는건 데이터셋 늘리는것뿐)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print("훈련셋: ", model.score(X_train, y_train))
# print("테스트셋: ", model.score(X_test, y_test))
# print("기울기: ", model.coef_)
# print("절편: ", model.intercept_)
# y_pred2 = X_test * model.coef_ + model.intercept_
# print(y_test)
# print(y_pred)
# print(y_pred2)
