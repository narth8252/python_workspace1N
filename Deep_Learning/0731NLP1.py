#250731 AM10시 쌤PPT 딥러닝종합_백현숙.ppt 433p
import string 
#print("구두점", string.punctuation)

class MyVectorize:
    def standardize(self, text): #표준화
        text = text.lower() #1.전부 소문자로 만든다
        return "".join(c for c in text if c not in string.punctuation)
        #구두점 제거한 문장을 만들어서 반환함
    
    def tokenize(self, text): #토큰화
        return text.split()   #잘라서 보낸다

    #어휘사전 만드는 함수
    def make_vocabulary(self, dataset):
        #1.전체 데이터셋을 순회하며 단어사전 만듬
        #2.기본적으로 빈문장""과 UNK토큰(unkown뭔지모르는말)
        #                     사람들이 일반적으로 고빈도사용하는 말만 차용하고, 그렇지않으면 언노운토큰으로 표현
        #3.새단어가 발견되면 어휘사전에 추가하고 해당단어에 숫자인덱스 부여
        self.make_vocabulary = {"":0, "[UNK]":1} #0과1은 특수목적으로 사용
        for text in dataset: #한문장씩 처리
            text = self.standardize(text) #표준화
            tokens = self.tokenize(text) #토큰화
            for token in tokens:
                if token not in self.vocabulary: #어휘사전에 없는 단어면 추가
                    self.vocabulary[token] = len(self.vocabulary)

        #역순서  단어:숫자 → 숫자:단어
        self.inverse_vocabulary = dict((v,k) for k,v in self.vocabulary.items())

     #문장을 받아서 벡터화
    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]
    
    def decode(self, int_sequence):
        return " ".join(self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)

mv = MyVectorize()
dataset=[
    "I write, erase, reqrite",
    "Erase again, and then",
    "A poppy blooms",
    "Dog is qute"
]

test = mv.standardize(dataset[0])
print(test)
test = mv.tokenize(test)
print(test)

mv.make_vocabulary(dataset)
print(mv.vocabulary)
print(mv.inverse_vocabulary)

print(mv.encode("I write erase"))
print(mv.decode([2,3,4,23])) #23은 UKN보여주기위해 쓰지않은 문자 인덱스 집어넣음