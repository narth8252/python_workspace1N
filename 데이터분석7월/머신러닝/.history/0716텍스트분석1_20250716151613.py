#텍스트분석: +-*/연산이 가능하게 변경
#sklearn이 CounterVectorizer 클래스 제공 -> RNN(순환신경망) 주로 사용
bards_words = ["I like star", "red star", "blue star", "I like dog"]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer() #객체만들기
vect.fit(bards_words) 