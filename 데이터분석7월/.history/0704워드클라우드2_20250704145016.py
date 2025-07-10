# pip install simplejson
import webbrowser
import pytagcloud
from konlpy.tag import Okt
from collections import Counter

file = open(".data/data1/txt", encoding="utf-8")
text = file.read()#

okt = Okt()
nouns = Okt.nouns(text) #명사로 분해하기
#파일로부터 명사추출
nounsCounter = Counter(nouns)
print(nounsCounter[:5])

tag = nounsCounter.most_common(100)
#모든단어로 차트그리면 너무 정신없어서
#빈도수를 기반으로 정렬한다음 100개만 가져와서 차트그려라

taglist = pytagcloud.make_tags(tag, maxsize=50)
print(taglist)