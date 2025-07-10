"""
1.데이터준비 일의80%차지 (전처리는 나중에는 본인이 직접) 이게 젤괴로움
  주업무80%: 데이터수집, 결측치처리, 이상치처리, 정규화, 주성분분석이나 차원축소 등,
  카테고리화 원핫인코딩 등
  아래 분석은 하다보면 같은코드로 하게돼있음.
2.데이터셋을 2개로 나눠, 훈련셋, 테스트셋으로 나눈다.
  (전부다 학습하면 과대적합인지 과소적합인지 미래예측력이 있는지 알수없어서,
  6:4 7:3 8:2 정도로 나눠서 테스트가능하도록, 훈련셋에만 맞추면안된다.
  일반화를 위해서 쪼개야한다.)
3. 알고리즘(Knn이웃 알고리즘,분류에서 가장 심플한 알고리즘)을 선택.
   분류알고리즘(로지스틱회귀분석, 서포트벡터머신, 의사결정트리,랜덤포레스트,그라디언트부스팅..)
   을 선택해 학습. 각 알고리즘마다 성능(학습더 잘하게) 올릴수있는 하이퍼파라미터가 있는데
   이걸 찾아내는 과정이 필요
4.예측
5.성능평가

"""
#myenv1서버에 프로그램 깔기
# cmd 관리자권한실행(anaconda됨) 
# conda activate myenv1
# conda install numpy scipy scikit-learn matplotlib ipython pandas imageio pillow graphviz python-graphviz
#깔리면 VS코드에도 
#conda activate myenv1
#VS코드 상단재생버튼(Python)으로 안되면 posershell에나 cmd에서
# 
#250707 am.10시 250701딥러닝_백현숙.PPT-269p 붓꽃..
# C:\Users\Admin\Documents\딥러닝2507  >  250701딥러닝_백현숙.PPT-p
# 저장폴더 C:\Users\Admin\Documents\GitHub\python_workspace1N\데이터분석250701\머신러닝250707
from sklearn.datasets import load_iris

data = load_iris() #Bunch라는 클래스타입
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

#scatter_matrix 차트가 직접 노가다로 그릴수도 이씨고 DataFrame
#sjavkdl qodufdmf -> DataFrame으로 바꾼다
import pandas as pd
iris_df = pd.DataFrame(X, columns=data['feature_names']) #numpy배열과 컬럼명으로
import matplotlib.pyplot as plt
#모든차트는 이거 꼭 필요(요즘은 시본차트 핫)

#PPT-269p.붓꽃 산포도그리기
pd.plotting.scatter_matrix( iris_df,
                           c=y, #각점의 색상지정. 0,1,2 각자 다른색
                           figsize=(15,15), #차트크기단위는 inch
                           marker='o',
                           hist_kwds={'bins':20}, #대각선의 히스토그램 구간개수
                           s=60, #점의 크기
                           alpha=0.8) #투명도, 1불투명, 0으로갈수록 투명
plt.show()

#250707 am11시 Knn이웃알고리즘
#이웃개수 지정가능. 대부분 홀수개 지정
#회귀분류 둘다가능
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3) #이웃개수3개, 바꿔가며 돌려봐야함.
#학습시작
#학습한내용은 모델자체가 갖고있고 충분히 모델의 하이퍼파라미터가 지정돼서
#최대한의 학습효과를 얻었다고 생각하면 모델 저장해놓고 나중에 불러와서 다시 쓸수있다.

model.fit(X_train, y_train)

#예측하기
y_pred = model.predict(X_test) #테스트셋으로 예측데이터 반환
#본래 테스트셋인 y_test와 비교
print(y_pred)
print(y_test)

#평가하기
print("훈련셋평가", model.score(X_train, y_train))
print("테스트셋평가", model.score(X_test, y_test))

#클래스 이름으로 출력
# class_names = list(data['target_names'])
# for i, j in zip(y_pred, y_test):
#     print("예측 :{20s} 실제 :{20s}".format(class_names[i], class_names[j]))

#로지스틱분류분석:2진,다중분류,R은 2진분류만. 0707_15시(딥러닝백현숙PPT-291p.)
#좋진않음.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 
model.fit(X_train, y_train)
print("<< 로지스틱분류분석 결과 >>")
print("훈련셋 점수 (R^2): ", model.score(X_train, y_train))
print("테스트셋 점수 (R^2): ", model.score(X_test, y_test))
#로지스틱도 가중치가 있음. 왜냐면 따라서 만들었으므로
print(model.coef_)
print(model.intercept_)

#0707 15:20 (딥러닝백현숙PPT-292p)
# 의사결정트리 (회귀와 분류분석 둘다 가능)
#필연적으로 과대적합된다. 알고리즘 자체가 과대적합으로 간다.
#의사결정트리 알고리즘은 특성의 중요도 파악용임
from sklearn.tree import DecisionTreeClassifier
#트리시작이 랜덤이라, 시드=파라미터 state=1 를 고정하지않으면 만들때마다 다른값 나옴.
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)
print("<< 의사결정트리 결과 >>")
print("훈련셋 점수(R^2): ", model.score(X_train, y_train))
print("테스트셋 점수(R^2): ", model.score(X_test, y_test))
print("특성의중요도: ", model.feature_importances_) #거의항상 과대적합임. 확인용

#수평막대차트 : 중요도
import matplotlib.pyplot as plt
import numpy as np
def treeChart(model, feature_name):
    #수평막대개수
    n_feature = len(model.feature_importances_)





