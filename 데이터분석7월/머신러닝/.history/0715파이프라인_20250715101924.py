#파이프라인 전체적으로 유기적 연결라인
#회귀평가와 분류평가: 머신러닝 모델이 갖고있는 score함수사용
#회귀score: 결정계수 R^2, 분류: 정확도기준, 불균형셋에 대해서는 이 기준 적용가능

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split# GridSearchCV
from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC #하이퍼파라미터 개수가많아서 해보자
from sklearn.metrics import classification_report,#accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


