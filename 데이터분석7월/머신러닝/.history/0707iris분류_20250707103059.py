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
#VS코드 상단재생버튼으로 
from sklearn.datasets import load_iris

data = load_iris() #Bunch라는 클래스타입
print(data.keys())

print("타겟이름", data['target_names'])
print("파일명", data['filename'])
print("데이터설명")
print(data["DESCR"])

#데이터 나누기
X = data["DESCR"]