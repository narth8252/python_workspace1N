from konlpy.tag import Okt
import re #정규식 모듈
from numpy import vectorize
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
# 벡터화: 자연여처리가 TextVectorization을 수정해서 내 코드 삽입

#1.한글형태소분석 모듈 객체생성
okt = Okt()

#2.어휘사전만들 텍스트
text = [
    "나는 AI 챗봇을 사용합니다.",
    "자연어 처리는 어렵습니다.",
    "챗봇과 대화하는 것은 문제해결을 못합니다.",
    "AI로 인해 사람들의 일자리가 사라지고 있습니다."
]
def clean_text(text):
    text = text.lower() # 대문자 → 소문자
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
    return text
    # 특수문자 제거 (한글, 영문, 숫자, 공백만 남김)
    # text = tf.strings.lower(text)
    # text = tf.strings.regex_replace(text, r"[^가-힣a-zA-Z0-9\s]", "")

#3.토큰나누기
def custom_standardation_fn(text):
    text = clean_text(text) #문자정리
    return " ".join(okt.morphs(text)) #토근화후 다시 합치기(토큰사이는 공백으로 구분)

#TextVectorization 객체 생성
vectorizer = TextVectorization(
    max_tokens = 1000, #TextVectorization 단어들을 토큰으로 분해해 숫자붙임.
    #자주쓰는 단어1000개를 숫서대로 어휘사전 만들겠다는 의미. 단어많으면 크기늘려
    output_mode = "int", #시퀀스로 만들라
    output_sequence_length = 20, #한문장에 쓰일 단어길이는 최대20개
    standardize = None  #외부에서 데이터를 변경해서 보내줄경우에 별도의 정규화함수 미사용
                  #None값 줘야함
)

# 3. Okt로 전처리된 데이터를 준비
#데이터셋을 사용해서 파일읽어 처리하려면 tensorflow가 제공하는 데이터타입으로 바꿔서 전달돼서
#파이썬에서 쓸수 있게 바꿔서 Okt적용하고나서 tensor타입으로 바꿔서 넣는다
data = [custom_standardation_fn(t) for t in text]
print("--- Okt로 전처리된 데이터 ---")
print(data)


# 4. TextVectorization 객체 생성
#adapt이후 어휘사전 확인
vectorizer = TextVectorization(
    max_tokens = 1000,
    output_mode = "int",
    output_sequence_length = 20,
    standardize = None,
    split = "whitespace"  # 공백 기준으로 토큰 분리
)

# 5. prepare_data로 준비된 데이터로 어휘사전 구축
# 여기서 'vectorizer'가 아닌 'vectorize'를 사용해야 합니다.
vectorizer.adapt(data)

# 6. adapt 이후 어휘사전 확인
vocabulary = vectorizer.get_vocabulary() #어휘사전 가져오기
print("--- 어휘 사전 (get_vocabulary) ---")
print(vocabulary)

# 7. 어휘사전을 단어:숫자 형태의 딕셔너리로 변환
word_to_index = {word: index for index, word in enumerate(vocabulary)}
print("--- 단어:숫자 형태의 어휘 사전 (딕셔너리) ---")
print(word_to_index)

# 8. 실제 텍스트를 벡터화하여 결과 확인
test_text = ["나는 자연어 처리를 배우고 있습니다."]
vectorized_text = vectorizer(test_text)
print("--- 테스트 문장 벡터화 결과 ---")
print(vectorized_text)
print("-" * 30)

vectorized_text = vectorizer([custom_standardation_fn("나는 자연어 처리를 배우고 있습니다."),
                             custom_standardation_fn("챗봇과 이야기를 합니다."),
                             custom_standardation_fn("자연어처리 중에 한글처리는 더 어려워요."),
                             ])
print(vectorized_text)