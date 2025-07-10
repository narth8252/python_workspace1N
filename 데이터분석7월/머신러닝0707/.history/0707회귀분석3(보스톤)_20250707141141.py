
# 250707 pm1시 250701딥러닝_백현숙.PPT-282p
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-p
# 저장폴더 C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707

# LinearRegression + KNN + 시각화 비교

import mglearn #
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 선형회귀 시각화

mglearn.plots.plot_linear_regression_wave()

plt.title("mglearn 선형회귀 예제")
plt.show()


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
