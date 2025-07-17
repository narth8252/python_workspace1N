#텍스트분석: +-*/연산이 가능하게 변경
#sklearn이 CounterVectorizer 클래스 제공 -> RNN(순환신경망) 주로 사용
# 텍스트를 수치 벡터로 바꾸기 → 머신러닝 모델이 처리할 수 있게
# CountVectorizer: 각 문장에서 단어의 출현 횟수를 기반으로 벡터화
bards_words = ["I like star", "red star", "blue star", "I like dog"]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer() #객체만들기
vect.fit(bards_words) #어휘 사전 생성

print("어휘사전 크기: ", len(vect.vocabulary_))
print("어휘사전 내용: ", vect.vocabulary_) #▶각 단어에 고유 번호(인덱스)를 부여한 것
# 어휘사전 크기:  5
# 어휘사전 내용:  {'like': 2, 'star': 4, 'red': 3, 'blue': 0, 'dog': 1}

# 수치 벡터로 변환
X = vect.transform(bards_words).toarray()
print(X)
# 행:문장, 열:단어('blue','dog','like','red', 'star') 순

값: 각 문장에서 해당 단어가 등장한 횟수
# [[0 0 1 0 1]   # "I like star"
#  [0 0 0 1 1]   # "red star"
#  [1 0 0 0 1]   # "blue star"
#  [0 1 1 0 0]]  # "I like dog"

응용: 단어 간 연산 가능