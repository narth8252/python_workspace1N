#250721 PM4시40 백현숙쌤 010-9083-7981
#쌤PPT118p (250717딥러닝종합_백현숙)

#로컬에 있는 문서를 읽어서 파싱하는것만 진행한다.


# 다음 스포츠 EPL 순위 데이터를 뷰티풀수프 없이 JSON 방식 API 호출로 받아오고, 파싱하여 pandas DataFrame에 담은 뒤 csv로 저장하려는 의도
from wsgiref import headers
import requests
import json
import pandas as pd
# https://finance.daum.net/api/sector/wics?perPage=5&order=desc

# API URL
url = "https://finance.daum.net/api/sector/wics?perPage=5&order=desc"
#권한없이 남의 사이트가서 퍼오면 안됨. 살짝 속이기 가능(그래도 막힐수있음)
#마치 내가 브라우저인것처럼 속이기
# 브라우저인 척 속이는 헤더
custom_header = {
    "referer": "https://finance.daum.net/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}
df = pd.DataFrame(columns=["changeRate", "sectorName", "name"])

text = ""
# 서버 요청
response = requests.get(url, headers=custom_header) 

if response.status_code == 200:
    text = json.loads(response.text) 
    # print(type(text))

    dataList = text["data"] #data:[{accTradeVolume: 27416973, accTradePrice: 17476623853, change: "RISE", sectorCode: "G151060",…}

    for item in dataList:
        data = dict()
        data["changeRate"] = item["changeRate"]               # %
        data["sectorName"] = item["sectorName"]       # 업종: 종이와목재
        data["name"] = item["includedStocks"][0]["name"]           # 국일제지
        df.loc[len(df)] = data


    # CSV로 저장 (Microsoft Excel 보기용)
    df.to_csv("국내주식업종상위.csv", encoding="utf-8-sig", index=False) #cp949인코딩은 엑셀은봐지는데, 메모장은 한글깨짐
    print(df)

# (deeplearning) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\Crawling>python 0721다음주식.py
#    changeRate sectorName    name
# 0    0.075542      종이와목재    국일제지
# 1    0.067449  에너지장비및서비스      태웅
# 2    0.040934         철강   TCC스틸
# 3    0.038820         조선    한화엔진
# 4    0.038803     복합유틸리티  지역난방공사