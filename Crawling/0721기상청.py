#250721 PM4시
#뷰티플스프를 이용해서 html파싱
#로컬에 있는 문서를 읽어서 파싱하는것만 진행한다.
#weather_table
import requests
#서버한테 정보를 보낼때 get, post 방식이 있다
url = "https://www.weather.go.kr/w/observation/land/city-obs.do"
response = requests.get(url)
if response.status_code == 200:
    text = response. text
    print(text)

#파싱하기
from bs4 import BeautifulSoup
import pandas as pd
doc = BeautifulSoup(text, "html.parser")
table = doc.find("table", {"id":"weather_table"})
trList = table.find_all("tr") #tr태그는 여러개라 find_all로 가져온다

df = pd.DataFrame(columns=["cityname","nowweather","temperature","humidity"])
i = 0
for tr in trList:
    # print(tr)
    mydict = {}   
    if len(tr.find_all("td")) > 0:
        th = tr.find("th")
        mydict["cityname"] = th.text
        tdList = tr.find_all("td")
        mydict["nowweather"]=tdList[0].text
        mydict["temperature"]=tdList[4].text
        mydict["humidity"]=tdList[8].text
        print(mydict)
        
        df.loc[len(df)] = mydict

print(df)





