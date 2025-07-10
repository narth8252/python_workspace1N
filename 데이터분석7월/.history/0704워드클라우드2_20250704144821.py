# pip install simplejson
import webbrowser
import pytagcloud
from konlpy.tag import Okt
from collections import Counter

file = open(".data/data1/txt", encoding="utf-8")
text = file.read()  

okt = Okt()
nouns = Okt.nouns(text) #명사로 분해하기

nounsCounter = Counter(nouns)
print()