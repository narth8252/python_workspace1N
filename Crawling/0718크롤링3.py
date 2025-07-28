#250718 쌤PPT 93,96,105p (250717딥러닝종합_백현숙)
#뷰티플스프 이용해서 html파싱
#로컬에 있는 문서 읽어서 파싱
html = open("./html/test1.html", encoding="utf-8")
doc = html.read()
html.close()

from bs4 import BeautifulSoup
from matplotlib import table
soup = BeautifulSoup(doc, "html.parser") #html은 DOM구조
#find("태그", "속성")함수 - 첫번째것만
#findAll("태그", "속성")함수 - 언제나 list형태, 인덱싱이나 for문으로 접근

#태그 객체가져오기
title_tag = soup.find("title") #<title>~</title>객체 통으로 가져옴
print(title_tag)
print("내용만", title_tag.text)

#h1태그 가져오기
h1 = soup.find("h1") #1st하나만 가져오기, 이렇게 안하면 통으로 들고옴
print(h1.text)

#h1태그 전체 가져오기
h1List = soup.find_all("h1") #언제나 list형태, 인덱싱이나 for문으로 접근
for h1 in h1List:
    print(h1.text)

#css selector, id랑 class이용하기
print("--- 태그와 id로 접근하기 ---")
hList = soup.find("h1", id="title1")
hList = soup.find("h1", {"id":"title1"}) #dict타입표기인데 윗줄도 어차피 dict으로 가지고오므로 동일하게 사용가능
for h1 in hList:
    print(h1.text)

print("--- 태그와 class로 접근하기 ---")
hList = soup.find("h1", class_="title1") #클래스는 언더바 써야함. 헷갈리니까 아래처럼 외워라
hList = soup.find_all("h1", {"class": "title1"})  # 또는 class_='title1'
for h1 in hList:
    print(h1.text)

print("--- ul 태그 가져오기 ---")
#1. ul태그 가져와서 이 태그로부터 li태그 리스트를 가져온다
ul = soup.find("ul",{"class":{"coffee"}})
print(ul) #못찾으면 None출력
liList = ul.find_all("li")
for li in liList:
    print(li.text)

print("--- 테이블 태그 가져오기 ---")
table = soup.find("table", id = "productList")
trList = table.find_all("tr")
for tr in trList:
    tdList = tr.find_all("td")
    for td in tdList:
        print(td.text, end=" ")
    print() 



# 왜 BeautifulSoup으로 HTML을 파싱할까요?
# 웹 페이지에서 원하는 정보를 효율적이고 체계적으로 추출하기 위해서입니다. 
# 웹 페이지는 사람이 보기에는 편리하지만, 컴퓨터가 데이터를 직접 사용하기에는 다소 복잡한 형태를 띠고 있어요.
# 1. 구조화된 데이터 추출 (스크레이핑/크롤링
# 웹 페이지는 대개 텍스트, 이미지, 링크 등 다양한 요소들이 복잡하게 얽혀 있는 **HTML(HyperText Markup Language)**이라는 문서 형식으로 구성되어 있습니다. 이 HTML은 마치 잘 정리된 책처럼 <head>, <body>, <div>, <p>, <a> 등 여러 태그로 이루어진 계층적인 구조(DOM 트리)를 가지고 있어요.
# BeautifulSoup은 이러한 HTML 문서를 파싱(parsing)하여, 사람이 쉽게 이해할 수 있는 객체 구조로 변환해 줍니다. 이렇게 되면 우리는 다음과 같은 작업을 쉽게 할 수 있습니다:
# 뉴스 기사의 제목, 본문, 작성일자만 추출하기
# 쇼핑몰 제품의 이름, 가격, 이미지 URL만 뽑아내기
# 게시판의 글 목록에서 각 글의 링크 주소와 제목 가져오기
# 특정 조건(예: 특정 클래스 이름을 가진 <div> 태그)에 맞는 요소들만 선택적으로 가져오기
# 이처럼 웹 페이지에서 필요한 데이터만 쏙쏙 골라내는 작업을 웹 스크레이핑(Web Scraping) 또는 **웹 크롤링(Web Crawling)**이라고 부릅니다. BeautifulSoup은 이러한 작업을 위한 강력하고 사용하기 쉬운 도구인 셈이죠.

# 2. 복잡한 HTML 문서 다루기 용이
# HTML 문서는 단순히 텍스트 파일을 읽는 것만으로는 원하는 데이터를 정확히 찾아내기 어렵습니다. 예를 들어, div 태그가 수십 개 있는데 그중 특정 id나 class를 가진 div 안에 있는 텍스트만 필요할 수 있죠.
# BeautifulSoup은 이러한 복잡한 시나리오에 대비해 find(), find_all(), select() 등 다양한 메서드를 제공합니다. 이를 통해 태그 이름, 속성(id, class 등), CSS 선택자 등 여러 조건을 조합하여 정확히 원하는 HTML 요소를 찾아낼 수 있습니다. 마치 도서관에서 원하는 책을 ISBN이나 저자명으로 정확하게 찾는 것과 같다고 할 수 있죠.

# 3. 데이터 분석 및 활용
# 추출된 데이터는 파이썬의 리스트나 딕셔너리 같은 형태로 다루기 쉬워지므로, 이후에 데이터 분석, 통계 처리, 데이터베이스 저장 등 다양한 목적으로 활용할 수 있게 됩니다.
# 요약하자면, BeautifulSoup을 이용한 HTML 파싱은 웹 페이지에서 필요한 정보를 체계적으로 추출하여 데이터를 수집하고 활용하기 위한 필수적인 과정이라고 할 수 있습니다.