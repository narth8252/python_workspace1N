#챗Gpt 검색 "pickle 모듈 알려줘"
#pickle은 파이썬 객체를 바이너리 형태로 직렬화(serialize)하거나 역직렬화(deserialize) 하는 데 사용되는 표준 라이브러리입니다. 
# 주로 데이터를 파일로 저장하거나 네트워크를 통해 전송할 때 사용합니다.
#바이너리(2차원)끼리, 텍스트끼리 성격같은것끼리만 돼서 피클로 저장한것만 읽어옴.
# 직렬화 (Serialization): 파이썬 객체 → 바이트 스트림
    #객체자체를 파일이나 네트워크로 메모리 그대로 저장.
# 역직렬화 (Deserialization): 바이트 스트림 → 파이썬 객체
    #파일이나 네트워크로부터 객체를 읽어들인다.

import pickle

#예시객체
data = {'name': '홍길동', 'age': 30, 
        'phone': ["010-000-0001", "010-000-0001", "010-000,0002"]}

#직렬화(serialize)객체를 파일로저장
with open('data.bin', 'wb') as f: #확장자 내맘 #피클쓰기
    pickle.dump(data, f) #dump로 보내고 딥러닝때도 별말없으면 피클씀.

#역직렬화(Deserialization)파일에서 객체읽기
with open('data.bin', 'rb') as f: #피클읽어오기
    data2 = pickle.load(f)

print(data2)
