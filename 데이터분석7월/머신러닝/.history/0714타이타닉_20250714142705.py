import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

#  1. 데이터 불러오기
# 사용자 지정 경로
path = r'C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석7월\data'

train_df = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
test_df = pd.read_csv(os.path.join(path, 'titanic_test.csv'))

print(train_df.shape)
print(train_df.head())

# 2. 데이터 전처리 함수
