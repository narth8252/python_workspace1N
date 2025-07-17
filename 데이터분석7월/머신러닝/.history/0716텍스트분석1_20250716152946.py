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
# 자주 등장하지만 중요하지 않은 단어의 가중치를 낮춤
# stop words 제거, sublinear_tf 조정도 가능

#  3. n-gram 설정: 연속된 단어 조합 반영
ngram_vect = CountVectorizer(ngram_range=(1, 2))  # unigram + bigram
X_ngram = ngram_vect.fit_transform(bards_words)

print("어휘 목록:", ngram_vect.get_feature_names_out())
print("n-gram 벡터:\n", X_ngram.toarray())
# 어휘 목록: ['blue' 'blue star' 'dog' 'i like' 'i like dog' 'like' 'like dog' ...]

# 4. 한국어 형태소 분석 + 벡터화
# 4-1. KoNLPy + Okt 이용
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer

okt = Okt()
kor_texts = ["나는 너를 좋아해", "너를 사랑해", "나는 강아지를 좋아해"]

def tokenize_okt(text):
    return okt.morphs(text)

vect_kor = CountVectorizer(tokenizer=tokenize_okt)
X_kor = vect_kor.fit_transform(kor_texts)

print("어휘 목록:", vect_kor.get_feature_names_out())
print("벡터:\n", X_kor.toarray())
# → 형태소 분석 결과를 기반으로 벡터를 만드니, 정확도가 더 올라감

# 5. 벡터를 RNN 입력용 sequence로 변환
# RNN이나 LSTM은 일반적으로 숫자 인덱스 시퀀스를 입력받음. Tokenizer를 쓰자.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

corpus = ["I like star", "red star", "blue star", "I like dog"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)
padded = pad_sequences(sequences, padding='post')

print("단어 인덱스:", tokenizer.word_index)
print("시퀀스:", sequences)
print("패딩된 시퀀스:\n", padded)
# 단어 인덱스: {'star': 1, 'i': 2, 'like': 3, 'red': 4, 'blue': 5, 'dog': 6}
# 시퀀스: [[2, 3, 1], [4, 1], [5, 1], [2, 3, 6]]
# 패딩된 시퀀스:
# [[2 3 1]
#  [4 1 0]
#  [5 1 0]
#  [2 3 6]]

# 정리 요약
#   
# CountVectorizer	단어개수기반 벡터	      기본 회귀/분류
# TfidfVectorizer	단어 중요도 반영	     SVM, Naive Bayes
# n-gram	        단어 조합 반영	        문맥 이해 향상
# 형태소 분석   	 한국어 처리 필수	      모든 모델
# Tokenizer+pad_sequences	시퀀스 벡터	    RNN, LSTM, GRU