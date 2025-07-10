from wordcloud import WordCloud
import nltk
from collections import Counter 
#한국 법률 말뭉치
from konlpy.corpus import kolaw
from konlpy.tag import  Okt
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import  ImageColorGenerator
from PIL import Image
import random

#토큰이 문장에서 단어를 하나씩 분리
#토큰나이저 - 문장에서 단어를 분리해내는 라이브러리
# 아래cmd창에 pip install nltk
nltk.download('punkt') #토큰나이저 다운코드
# nltk.download('punkt_tab') #토큰나이저 다운코드
from nltk.tokenize import sent_tokenize #문장으로 쪼개주는 토큰나이즈
from nltk.tokenize import word_tokenize
c = open("./data1/data/alice.txt", "r").read()
token_en = word_tokenize
print()


# #한국 법률 말뭉치
# from konlpy.corpus import kolawd
# c = kolaw.open('constitution.txt').read()
# print(c[:200])
# fontpath = "C:\Windows\Fonts\나눔고딕.ttf"
# wordcloud = WordCloud(font_path=fontpath).generate(c)
# #파일로저장
# wordcloud.to_file("image1.png")

# 이미지정보를 리턴
# plt.imshow(wordcloud, interpolation='bilinear') #이미지 이뻐보이게 보간법
# plt.axis("off") #x,y축 안보이게
# plt.show()
# file = open("./data1/data/alice.txt")
# text = file.read()
# #()넣어야 객체만들어짐.
# wordcloud = WordCloud().generate(text)
# #이미지정보를 리턴


