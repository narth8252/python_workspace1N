import matplotlib.pyplot as plt
from wordcloud import WordCloud

file = open("./data1/data/alice.txt")
text = file.read()

#()넣어야 객체만들어짐.
wordcloud = WordCloud().generate(text)

#이미지정보를 리턴
plt.imshow(wordcloud, interpolation='bilinear') #이미지 이뻐보이게 보간법
plt.axis("off") #x,y축 안보이게
plt.show()