import matplotlib.pyplot as plt
from wordcloud import WordCloud
# from konlpy.tag import Okt
#plt 라이브러리가 한글지원안한다. 한글쓰려면 폰트지정해야함.



#한국 법률 말뭉치
from konlpy.corpus import kolawd
c = kolaw.open('constitution.txt').read()
print(c[:200])
wordcloud = WordCloud(font_path=font).generate(text)

# file = open("./data1/data/alice.txt")
# text = file.read()
# #()넣어야 객체만들어짐.
# wordcloud = WordCloud().generate(text)
# #이미지정보를 리턴
# plt.imshow(wordcloud, interpolation='bilinear') #이미지 이뻐보이게 보간법
# plt.axis("off") #x,y축 안보이게
# plt.show()