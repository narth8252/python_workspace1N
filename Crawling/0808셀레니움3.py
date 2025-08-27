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
driver = webdriver.Chrome(options = chrome_options) #드라이버켜짐
#driver 객체가 브라우저 객체임
driver.implicitly_wait(3) #페이지열릴때까지 page없어도 3초기다림
driver.get("http://www.python.org")
assert "Python" in driver.title #assert 디버깅툴
input_box = driver.find_element(By.NAME, "q") #구글창 F12해서 찾은게 name=q였음(name속성이용해서 찾겠다.)

time.sleep(2)
#input_box에 키캆보냄
input_box.send_keys("python") #키보드이벤트 발생
time.sleep(3)

# 1.input_box.submit() #서버로 정보전송
#input_box.submit()  # 또는 input_box.send_keys(Keys.RETURN) 중 택1
#서버로 전송해서 작동하게하려면 자바스크립트에서 submit 이라는게 필요함
#개발자가 가끔 까먹는경우 있어서 엔터키누른다고 submit 자동호출되지않음
#웹개발자들이 커서있는곳에서 엔터키(RETURN)누르면 서버로 전송
# 2.input_box.send_keys(Keys.RETURN) #옛날키보드는 엔터키가 RETURN이라고 써있었음.그래서 Enter의미임
# 셀레니움에서는 Keys라는 클래스 이름을 사용
#리턴은 안갈수도 있어서 submit이 확실함
#엔터키 누르면 submit함수 호출되게 해놨을때 동작함
# 3.submit 버튼 찾아서 이벤트 발생
#파이선홈피는 F12해보니 go 버튼 찾아오려고 보니 name, submit, 다 있음
button = driver.find_element(By.NAME, "submit")
button.click() #버튼눌림 이벤트 → click이벤트 핸들러가 submit()호출

print(driver.current_url) #현재문서 url
print(driver.title)       #현재문서 제목
#driver.page_source #현재문서

driver.set_window_size(400,400)
driver.save_screenshot("python_event.png") #화면캡쳐 후 저장

#다음페이지로 이동하기
#content > div > section > form > div > a:nth-child(1)
#content > div > section > form > div > a
next = driver.find_element(By.CSS_SELECTOR, "#content > div > section > form > div > a:nth-child(1)")
next.click()

time.sleep(5)
# //*[@id="content"]/div/section/form/div/a[2]

driver.quit()