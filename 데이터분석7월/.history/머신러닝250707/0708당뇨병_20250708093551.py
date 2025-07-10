#250708 am9:20 
#당뇨병과 관련된 요소들이 있음, 1년뒤 값 예측
#알고리즘 중에 Knn이웃,의사결정트리,랜덤포레스트 등 몇개는 분류,회귀 모두 지원
#Lasso, Ridge
from sklearn.datasets import load_diabetes
#bunch라는 클래스타입으로 정리해서 주고, 
# 이상치,누락치,정규화까지 된 자료를 줌(이게 우리주업무) -pandas, numpy
data = load_diabetes()

#로지스틱회귀분류
print(data.keys())
print(data['target'][:10]) #[:10]10개만출력
print(data['data'][:10])
print(data['DESCR'])

X = data["data"]  #ndarray2차원배열, 현재10개의 특성값이
y = data["target"]#ndarray1차원배열, 미래예측값으로 나타남

print(X.shape) #데이터442개, 특성10개
print(y.shape)

#데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234) 
#튜플4개, 같은비율로 잘라내려고, test size=안주면 7.5:2.5비율로 자동나눔
#한줄넘어가서 엔터쳤더니 빨간줄 뜨면 역슬래시\
print(X_train.shape)
print(X_test.shape)
