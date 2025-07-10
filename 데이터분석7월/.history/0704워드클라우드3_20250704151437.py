import matplotlib.pyplot as plt
from wordcloud import WordCloud

file = open("./data1/data/alice.txt")
text = file.read()

wordcloud = WordCloud.generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()