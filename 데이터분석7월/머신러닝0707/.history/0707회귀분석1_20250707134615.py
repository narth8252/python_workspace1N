#250707 pm1시 250701딥러닝_백현숙.PPT-282p
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-p
# 저장폴더 C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707

#데이터 분석 
#사이킷런 - 입력데이터, 2d tensor, 출력 1d tensor 

# (특성 달랑1개. 개개인의 차이(IQ,1~5차모의고사 성적)같은거 넣으면 특성늘어남. )
#공부시간
x = [[20], [19], [17], [18], [12], [14], [10], [9], [16], [6]]
#평균값 
y = [100,100, 90, 90, 60, 70, 40, 40, 70, 30]

import numpy as np
X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression
model = LinearRegression() #하이퍼파라미터없음(과대,과소던 할수있는건 데이터셋 늘리는것뿐)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("훈련셋: ", model.score(X_train, y_train))
print("테스트셋: ", model.score(X_test, y_test))
print("기울기: ", model.coef_)
print("절편: ", model.intercept_)
y_pred2 = X_test * model.coef_ + model.intercept_
print(y_test)
print(y_pred)
print(y_pred2)
#다중회귀분석은 가중치 많다. 각 독립변수마다 별도의 가중치 가져온다.
#위에 공부시간,평균값은 조작한 데이터라 깔끔하게 결과 나옴.

# w1x1 + w2x2 + w3x3 ..... + wnxn
"""
(w1,w2,w3,...wn) x (x1,
                    x2,
                    x3,
                    x4,

"""