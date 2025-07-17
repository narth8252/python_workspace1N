#250716 AM9:20
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic' #내컴파이썬에 설치된 폰트명으로 변경
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # 특성 스케일링 (선형 모델에 중요)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer # 유방암 데이터셋

 
cancer = load_breast_cancer()
X = pd.DateFrame(cancer.data, columns= cancer.feature_names)
y = cancer.target
