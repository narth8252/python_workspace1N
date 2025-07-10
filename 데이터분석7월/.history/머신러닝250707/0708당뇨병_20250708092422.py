#250708 am9:20 
#당뇨병과 관련된 요소들이 있음, 1년뒤 값 예측
#알고리즘 중에 Knn이웃,의사결정트리,랜덤포레스트 등 몇개는 분류,회귀 모두 지원
#Lasso, Ridge
from sklearn.datasets import load_diabetes
#bunch라는 클래스타입으로 정리해서 주고, 
# 이상치,누락치,정규화까지 된 자료를 줌(이게 우리주업무) -pandas, numpy
data = load_diabetes()

print(data.keys())
print(data['target'])
print(data['data'])


