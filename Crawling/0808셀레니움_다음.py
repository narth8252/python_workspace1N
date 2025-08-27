#250808 AM 9시 웹크롤링 복습
#conda install selenium cmd관리자로 (base)에 설치.
#이전에는 드라이브파일을 다운받아서 연결하는 방식
#캡쳐하기
from selenium import webdriver
from selenium.webdriver.chrome.options import Options #브라우저를 처음 작동시킬때 옵션을 줄 수 있다
from selenium.webdriver. common.by import By # 857
from selenium.webdriver.common.keys import Keys # 키값제어 라이브러리(Keys.RETURN, Keys.ENTER, Keys.TAB 같은 키보드 입력을 쓸 수 있어.)
from bs4 import BeautifulSoup
import time

#1.Options객체를 만들어서 크롬드라이버의 기본 설정을 관리한다
#option매우많으니 AI물어봐서 찾아서 해라
chrome_options = Options()
# chrome_options.add_argument("--window-size=1920,1000") #윈도우 크기지정
chrome_options.headless = False 
#브라우저키고 작업완료시 자동닫힘 방지 옵션
chrome_options.add_experimental_option("detach", True) 
#detach 분리, attech 접속

#크롬 드라이버 실행후 옵션적용
driver = webdriver.Chrome(options = chrome_options)
#driver객체가 브라우저를 객체임
driver.implicitly_wait(3) #페이지가 열릴때까지 3초쯤 기다려라
driver.get("https://www.daum.net/")

input_box = driver.find_element(By.NAME, "q") #내가 name속성을 이용해서 찾을 거다
#name이 q였음

time.sleep(2) #프로세스가 cpu를 2초간 뱃긴다
#input_box에 키값을 보냄
input_box.send_keys("말복") #키보드 이벤트를 발생시킴
input_box.submit() #서버로 정보를 전송

#뉴스탭
#daumGnb > div.tab_dynamic.tab_flex > ul > li:nth-child(2) > a
btnNews = driver.find_element(By.CSS_SELECTOR,
                              "#daumGnb > div.tab_dynamic.tab_flex > ul > li:nth-child(2) > a")

btnNews.click()

#F12 해서 1p부터 여러페이지 봐가면서 반복성 찾아라
#dnsColl > div:nth-child(2) > div > div > a:nth-child(2)
#dnsColl > div:nth-child(2) > div > div > a:nth-child(3)
#dnsColl > div:nth-child(2) > div > div > a:nth-child(4)
#dnsColl > div:nth-child(2) > div > div > a:nth-child(5)
#폰은 비행기모드 껏다키면 ip가 새로돼서 크롤링하기 편함 - 웹은 계속 들어가면 공격으로 인식해서 며칠있으면 짤림.
import random
i = 0
while True: #무한루프
    time.sleep(random.randint(1,12))
    print(f"{i} page ....")

    doc = BeautifulSoup(driver.page_source)
    ul = doc.find("ul", {"class":"c-list-basic"})
    liList = ul.find_all("li")
    for li in liList:
        print(li.text)
        # title = li.find("span", {"class":"txt_info"})
        # print(title.text)
    
    #다음페이지 이동
    i = i+1
    next = driver.find_element(By.CSS_SELECTOR, f"#dnsColl > div:nth-child(2) > div > div > a:nth-child({i})")
    if next == None or i>5: #5개까지만 돌려보고 작동하면 전체코딩(첨부터 끝까지 다돌리다가 큰일나는수있음)
        break
    next.click() 
#driver.quit() #종료


# 셀레니움을 안 쓸 경우 웹 자동화나 웹 데이터 수집(Crawling/Scraping)에서 다음과 같은 **제약이나 문제점**이 생긴다:
# ## ✅ 1. **JavaScript 기반 동적 페이지 처리 불가**
# * 대부분의 웹사이트는 **JavaScript로 콘텐츠를 동적으로 렌더링**함.
# * `requests`, `BeautifulSoup` 같은 정적 크롤링 도구는 **HTML이 로딩되기 전의 정적 코드**만 가져옴.
# * 예:
#   * 뉴스 웹사이트에서 스크롤해야 기사 목록이 로드됨 (infinite scroll)
#   * 로그인 이후에만 보이는 사용자 정보
# 📌 셀레니움은 실제 브라우저를 띄워 JS 실행까지 해주므로 동적 웹 페이지도 처리 가능함.
# ---
# ## ✅ 2. **로그인/클릭/스크롤 등 사용자 행동 자동화 불가**
# * `requests`는 HTML만 받아오므로 다음 작업 불가능:
#   * 버튼 클릭
#   * 드롭다운 조작
#   * 입력창에 키보드 입력
#   * 마우스 오버 등
# 📌 셀레니움은 실제 사람처럼 **브라우저를 조작**함. 로그인 자동화, 게시글 작성, 검색어 입력 등 가능.
# ---
# ## ✅ 3. **봇 탐지 우회 어려움**
# * 일부 웹사이트는 `requests`나 `urllib` 등 \*\*비브라우저 요청(User-Agent가 없음)\*\*을 차단함.
# * Cloudflare, Akamai 등 **보안 기능 강화된 사이트**에서는 headers 조작만으로는 한계 있음.
# 📌 셀레니움은 진짜 브라우저를 쓰기 때문에 **봇 탐지 우회가 상대적으로 쉬움**.
# ---
# ## ✅ 4. **에러 디버깅이 어려움**
# * `requests`, `BeautifulSoup`은 화면이 없기 때문에, 로딩이 안되거나 구조가 바뀌었을 때 원인 파악이 어려움.
# 📌 셀레니움은 크롬 화면이 실제 뜨니까, 어떤 DOM이 보이는지 직접 눈으로 확인하며 디버깅 가능.
# ---
# ## ✅ 5. **복잡한 상호작용 기반의 작업 불가능**
# * 예: 쇼핑몰 장바구니 담기 → 결제창 이동 → 정보 입력 → 결제 버튼 클릭
# * 이런 시나리오는 HTTP요청만으로는 구현 거의 불가능.
# 📌 셀레니움은 이러한 **복잡한 플로우도 자동화 가능**함.
# ---
# ## 🔻 요약
# | 기능/도구        | requests/BeautifulSoup | Selenium |
# | ------------ | ---------------------- | -------- |
# | 정적 HTML 수집   | ✅                      | ✅        |
# | 동적 JS 콘텐츠 처리 | ❌                      | ✅        |
# | 로그인 자동화      | ❌                      | ✅        |
# | 클릭/스크롤       | ❌                      | ✅        |
# | 봇 우회         | ❌                      | ✅        |
# | 디버깅 용이성      | ❌                      | ✅        |
# ---
# 필요에 따라 `requests + BeautifulSoup` → 빠르고 가볍고
# 복잡한 자동화가 필요한 경우엔 `selenium`을 꼭 써야 함.
