#myenv1서버에 프로그램 깔기
# cmd 관리자권한실행(anaconda됨) 
# conda activate myenv1
# conda install numpy scipy scikit-learn matplotlib ipython pandas imageio pillow graphviz python-graphviz
#깔리면 VS코드에도 
#conda activate myenv1
#VS코드 상단재생버튼(Python)으로 안되면 posershell에나 cmd에서
# 
#250707 am.10시 250701딥러닝_백현숙.PPT-289p 암환자(로지스틱)-2진분류..
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-p
# 저장폴더 C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer() #Bunch라는 클래스타입
print(data.keys())

print("타겟이름", data['target_names'])
print("파일명", data['filename'])
print("데이터설명")
print(data["DESCR"])

#데이터 나누기
X = data["data"]  #ndarray 2차원배열
y = data["target"]#ndarray 1차원배열

print(X[:10])
print(y)

#데이터 랜덤섞어서 70%추출
from sklearn.model_selection import train_test_split
#tuple로 반환, random_state인자가 seed역할, 
#계속 같은데이터내보내려면 이값고정, 얘바꾸면 데이터바뀜
#test_size=0.3 그 비율대로 나눈다.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)
print(X_train.shape)
print(X_test.shape)

#데이터 전체 확인하기 위해서, 산점행렬(특성4개면 각 틍성으로만 그릴수 있어서 차트4x4=16개)
#특성 10roehlaus 10x10 = 100개 차트 만들어진다.

# # scatter_matrix 차트가 직접 노가다로 그릴수도 이씨고 DataFrame
# # sjavkdl qodufdmf -> DataFrame으로 바꾼다
# import pandas as pd
# iris_df = pd.DataFrame(X, columns=data['feature_names']) #numpy배열과 컬럼명으로
# import matplotlib.pyplot as plt
# # #모든차트는 이거 꼭 필요(요즘은 시본차트 핫)

# #유방암 분류는 산포도 너무 오래걸려서 비추
# pd.plotting.scatter_matrix( iris_df,
#                            c=y, #각점의 색상지정. 0,1,2 각자 다른색
#                            figsize=(15,15), #차트크기단위는 inch
#                            marker='o',
#                            hist_kwds={'bins':20}, #대각선의 히스토그램 구간개수
#                            s=60, #점의 크기
#                            alpha=0.8) #투명도, 1불투명, 0으로갈수록 투명
# plt.show()

# #250707 am11시
# #Knn이웃알고리즘

#이웃개수 지정가능. 대부분 홀수개 지정
#회귀분류 둘다가능
from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=3) #이웃개수3개, 바꿔가며 돌려봐야함.
#학습시작
#학습한내용은 모델자체가 갖고있고 충분히 모델의 하이퍼파라미터가 지정돼서
#최대한의 학습효과를 얻었다고 생각하면 모델 저장해놓고 나중에 불러와서 다시 쓸수있다.

# model.fit(X_train, y_train)

# #예측하기
# y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
# #본래 테스트셋인 y_test와 비교
# print(y_pred)
# print(y_test)

#평가하기
#모델평가시 주의사항: 예측률높은게 좋은모델
#모델자체는 우수한데 암환자인데 암환자 아니라고 판단내렸을때 벌어질일, 모델자체는 80%밖에 안나옴.
#그런데 모든 암환자를 다 찝어내는데 암환자가 아닌사람을 암환자로 인식하는경우
#적중률 떨어지더라도 많은사
# print("훈련셋평가", model.score(X_train, y_train))
# print("테스트셋평가", model.score(X_test, y_test))

n_neighbors = 10 #적당히
trainscoreList=list()
testscoreList=list()
for i in range(1, n_neighbors+1):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    score1 = model.score(X_train, y_train)
    score2 = model.score(X_test, y_test)
    trainscoreList.append(score1)
    testscoreList.append(score2)

# scatter_matrix 차트가 직접 노가다로 그릴수도 있고 DataFrame
# sjavkdl qodufdmf -> DataFrame으로 바꾼다
# import pandas as pd
# iris_df = pd.DataFrame(X, columns=data['feature_names']) #numpy배열과 컬럼명으로
import matplotlib.pyplot as plt
# #모든차트는 이거 꼭 필요(요즘은 시본차트 핫)
# x축,y축
plt.plot(ra)


# #클래스 이름으로 출력
# class_names = list(data['target_names'])
# for i, j in zip(y_pred, y_test):
#     print("예측 :{20s} 실제 :{20s}")
