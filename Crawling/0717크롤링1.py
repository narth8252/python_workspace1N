#250717 PM4시
#쌤PPT103p (250717딥러닝종합_백현숙)

#웹스크래핑, 크롤링, 웹사이트마다 데이터를 갖고오는 방식이 다르다
#계속 웹사이트가 업그레이드가 되어서, 너무 많이들하니까 막는 기술도 있긴
#리캡챠

#urllib 옛날 == > requests 라는 모듈이 있음
#1. requests 모듈에 get방식, post방식으로 웹서버랑 접속해서 문서를 가져올 수 있다
#문서, 이미지, 파일
#서버쪽에서 보내는 응답을 받아온다. html(일반웹서버), json형태임(restpul api서버)
#html -> 이 문서를 분석해서 데이터만 추출(파싱이라고 한다)
#html1문서로부터 데이터를 파싱하는 알고리즘 BeautifulSoup - 설치를 해야한다
#json라이브러리

#셀레니움 - 크롬을 만들다가 디버깅용 툴을 만들음, 사용이 어려움 웬만하면 된다.
#        - 이벤트나 자바스크립트 호출이 가능희

# 1. requests 모듈 설치 (요청 ↔ 응답)
# 2. BeautifulSoup 모듈 설치
# 3. requests로 웹서버에 접속해서 문서를 가져온다
# 4. BeautifulSoup로 문서를 파싱해서 원하는 데이터 추출
# 5. 추출한 데이터를 원하는 형태로 가공
# 6. 가공한 데이터를 저장하거나 출력
# 7. 필요시 반복문을 사용하여 여러 페이지에서 데이터 추출
# 8. 크롤링한 데이터를 분석하거나 시각화
# 9. 크롤링한 데이터를 데이터베이스나 파일로 저장
# 10. 크롤링한 데이터를 활용하여 머신러닝 모델 학습 등 다양한 작업 수행

import requests
from bs4 import BeautifulSoup
#pip install requests beautifulsoup4

# 1. requests 모듈을 사용하여 웹 페이지 가져오기
response = requests.get('https://www.pythonscraping.com/exercises/exercise1.html')
print("응답 상태 코드:", response.status_code)  # 200이면 성공

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'lxml')
    # 나머지 코드 전부 들여쓰기해서 이 조건문 안으로 넣는 게 안전

    #response.text  # 응답 내용 출력
    #response.content  #응답 내용 출력 (바이너리 형식), 이미지나 동영상처리

print("응답 내용:", response.text)  # HTML 문서 내용 출력

# 2. HTML 파싱 및 요소 추출
# 2-1. BeautifulSoup을 사용하여 HTML 문서 파싱
# 'html.parser'도 괜찮지만, lxml 설치되어 있으니 더 빠른 파싱기로 바꿔도 OK
# BeautifulSoup(response.text, 'html.parser')
soup = BeautifulSoup(response.text, 'lxml')  # lxml 파서 사용

# 2-2. 원하는 데이터 추출
title = soup.title.string  # <title> 태그의 문자열 추출
print("페이지 제목:", title)  # 페이지 제목 출력
# 2-3. 특정 요소 찾기
heading = soup.find('h1')  # <h1> 태그 찾기
print("헤딩 내용:", heading.text)  # 헤딩 내용 출력
# 2-4. 모든 <p> 태그 찾기
paragraphs = soup.find_all('p')  # 모든 <p> 태그 찾기
for i, p in enumerate(paragraphs):
    print(f"문단 {i+1} 내용:", p.text)  # 각 문단 내용 출력
# 6. 특정 클래스나 아이디를 가진 요소 찾기
special_element = soup.find(class_='special')  # 클래스가 'special'인 요소 찾기
if special_element:
    print("특별한 요소 내용:", special_element.text)  # 특별한 요소 내용 출력
# 7. 링크 추출
links = soup.find_all('a')  # 모든 <a> 태그 찾기
for link in links:
    href = link.get('href')  # 링크의 href 속성 추출
    if href:
        print("링크:", href)  # 링크 출력
# 8. 이미지 추출
images = soup.find_all('img')  # 모든 <img> 태그 찾기
for img in images:
    src = img.get('src')  # 이미지의 src 속성 추출
    if src:
        print("이미지 링크:", src)  # 이미지 링크 출력
# 9. 크롤링한 데이터를 데이터베이스나 파일로 저장 (예: SQLite, CSV 등 사용)
# 10. 반복문을 사용하여 여러 페이지에서 데이터 추출 (예: 페이지 번호를 변경하여 반복)
# 11. 크롤링한 데이터를 분석하거나 시각화 (예: Matplotlib, Pandas 등 사용)
# 13. 크롤링한 데이터를 활용하여 머신러닝 모델 학습 및 예측
# 14. 크롤링한 데이터를 활용하여 웹 애플리케이션 개발 등 다양한 작업 수행

