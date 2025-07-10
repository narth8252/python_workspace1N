import matplotlib.pyplot as mp
from workcloud

file = open("./data1/data/alice.txt")
text = file.read()

wordcloud = WordCloud.generate(text)

plu