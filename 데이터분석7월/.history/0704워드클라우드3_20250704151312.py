import matplotlib.pyplot as mlt
from workcloud

file = open("./data1/data/alice.txt")
text = file.read()

wordcloud = WordCloud.generate(text)

plt.imshow