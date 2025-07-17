#250716 AM9:20
import pandas as pd
import numpy as np
import optuna
from sklearn.datasets import fetch_california_housing #"회귀Regression문제" 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. California Housing 데이터셋 로드
print("----- California Housing 데이터셋 로드 중 -----")
housing = fetch_california_housing()
print(type(housing)) #딥러닝(Tensorflow:C++) → 케라스(초보자용),파이토치(세밀제어가능)

#차트그리려면 numpy → 요소하나씩 그려야해서 차트그리고싶다.
#DataFrame: 데이타프레임자체가 차트제공하기도 하고, seaborn, 
X = 
