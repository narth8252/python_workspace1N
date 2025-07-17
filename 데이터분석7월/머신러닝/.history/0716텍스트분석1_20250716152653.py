#250716 PM3시  텍스트 벡터화 전체 흐름
# 텍스트분석: +-*/연산이 가능하게 변경
#sklearn이 CounterVectorizer 클래스 제공 -> RNN(순환신경망) 주로 사용
# 텍스트를 수치 벡터로 바꾸기 → 머신러닝 모델이 처리할 수 있게
# CountVectorizer: 각 문장에서 단어의 출현 횟수를 기반으로 벡터화

# 샘플문장
bards_words = ["I like star", "red star", "blue star", "I like dog"]

# 1. CountVectorizer: 단어 개수 기반 벡터화
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer() #객체만들기
vect.fit(bards_words) #어휘 사전 생성
X = vect.fit_transform(bards_words)

print("어휘사전 크기: ", len(vect.vocabulary_))
print("어휘사전 내용: ", vect.vocabulary_) #▶각 단어에 고유 번호(인덱스)를 부여한 것
print("어휘 목록:", vect.get_feature_names_out())
print("벡터:\n", X.toarray())
# 어휘사전 크기:  5
# 어휘사전 내용:  {'like': 2, 'star': 4, 'red': 3, 'blue': 0, 'dog': 1}
# 어휘 목록: ['blue' 'dog' 'like' 'red' 'star']
# 벡터:
#  [[0 0 1 0 1]
#   [0 0 0 1 1]
#   [1 0 0 0 1]
#   [0 1 1 0 0]]

# 수치 벡터로 변환
X = vect.transform(bards_words).toarray()
print(X)
#행:문장, 열:단어('blue','dog','like','red','star')순, 값:각문장에서 해당단어 등장횟수
# [[0 0 1 0 1]   # "I like star"
#  [0 0 0 1 1]   # "red star"
#  [1 0 0 0 1]   # "blue star"
#  [0 1 1 0 0]]  # "I like dog"

# 응용: 단어 간 연산 가능: 수학 연산도 가능
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(X)
print(similarity)
# → 문장 간 유사도 계산, 군집화, RNN/LSTM 입력값 등 다양하게 활용 가능

# 1. transform() → 벡터 결과 보기
X = vect.transform(bards_words)
print(X.toarray())  # 희소행렬을 밀집행렬로 보기 좋게

# 2. 벡터화된 단어 순서 확인
print(vect.get_feature_names_out())  # ['blue', 'dog', 'like', 'red', 'star']

# 3. 다른 벡터라이저 실험
# • TfidfVectorizer(): 단어의 중요도 반영
# • HashingVectorizer(): 어휘 사전 없이 해시 기반
# 2. TfidfVectorizer: 단어 중요도(TF-IDF) 반영
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
X_tfidf = tfidf_vect.fit_transform(bards_words)

print("어휘 목록:", tfidf_vect.get_feature_names_out())
print("TF-IDF 벡터:\n", X_tfidf.toarray())
자주 등장하지만 중요하지 않은 단어의 가중치를 낮춤

stop words 제거, sublinear_tf 조정도 가능