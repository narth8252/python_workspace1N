#250721 PM4시40 백현숙쌤 010-9083-7981
#쌤PPT117p (250717딥러닝종합_백현숙)

#로컬에 있는 문서를 읽어서 파싱하는것만 진행한다.

#카카오 디벨로퍼
# https://developers.kakao.com/tool/rest-api/open/get/v2-user-me

# 다음 스포츠 EPL 순위 데이터를 뷰티풀수프 없이 JSON 방식 API 호출로 받아오고, 파싱하여 pandas DataFrame에 담은 뒤 csv로 저장하려는 의도
from wsgiref import headers
import requests
import json
import pandas as pd
# https://finance.daum.net/api/search/ranks?limit=10

#서버한테 정보를 보낼때 get, post 방식이 있다
# url = "https://sports.daum.net/record/epl" #이렇게해도 못읽어옴
#아래처럼 해야함
# https://sports.daum.net/record/epl
# NAME에 rank.json?leagueCode=어쩌고 더블클릭 해서 새창에 뭐 엄청 써진거 열리면 그나마 오픈된것임.
# Headers에 제너럴에 Request URL 복붙해와야 읽을수있음
# API URL
url = "https://sports.daum.net/prx/hermes/api/team/rank.json?leagueCode=epl&seasonKey=20252026&page=1&pageSize=100"
#권한없이 남의 사이트가서 퍼오면 안됨. 살짝 속이기 가능(그래도 막힐수있음)
#마치 내가 브라우저인것처럼 속이기
# 브라우저인 척 속이는 헤더
custom_header = {
    "referer": "https://sports.daum.net/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}
df = pd.DataFrame(columns=["name", "nameMain", "gf"])

text = ""
# 서버 요청
response = requests.get(url, headers=custom_header) #그래도 못가져오면 api호출

if response.status_code == 200:
    text = json.loads(response.text) #json.loads()함수로 dict타입으로 변경
    # print(type(text))

    dataList = text["list"]
    # 결과 누적 리스트
    results = []
    for item in dataList:
        data = dict()
        data["name"] = item["name"]               # 팀 이름
        # data["gp"] = item["stat"]["gp"]           # 경기 수
        # data["win"] = item["stat"]["win"]         # 승
        data["nameMain"] = item["nameMain"]       # 무
        # data["loss"] = item["stat"]["loss"]       # 패
        data["gf"] = item["rank"]["gf"]           # 득점
        # data["ga"] = item["stat"]["ga"]           # 실점
        # data["gd"] = item["stat"]["gd"]           # 득실차
        df.loc[len(df)] = data
        results.append(data)

    # pandas DataFrame 변환
    df = pd.DataFrame(results)

    # CSV로 저장 (Microsoft Excel 보기용)
    df.to_csv("해외축구.csv", encoding="utf-8-sig", index=False) #cp949인코딩은 엑셀은봐지는데, 메모장은 한글깨짐
    print(df)


# 크롤링보다는 이렇게 **공식 API(JSON)**이 제공되는 경우, requests + json이 더 정확하고 안정적입니다.
# JSON 구조를 파악하려면 print(json.dumps(text, indent=2, ensure_ascii=False))로 보면 좋습니다.
# pandas를 응용하여 정렬, 필터, 시각화도 쉽게 할 수 있습니다.

# ✅ 결론
# ✔️ 딥러닝이나 머신러닝에 활용할 수 있는 형태의 외부축구 데이터를 자동으로 수집 및 저장하는 코드가 완성되었습니다.
# 엑셀에서 열어볼 수 있고, 나중에 분석 단계에서 바로 데이터프레임 불러와 활용 가능합니다.

# (deeplearning) C:\Users\Admin\Documents\GitHub\python_workspace1N\Data_Analysis_2507\Crawling>python 0721해 외축구.py
#                  name           nameMain  gf
# 0    브라이튼 앤 호브 알비온 FC   브라이튼 앤 호브 알비온 FC   0
# 1             AFC 본머스            AFC 본머스   0
# 2         노팅엄 포레스트 FC        노팅엄 포레스트 FC   0
# 3           뉴캐슬 유나이티드          뉴캐슬 유나이티드   0
# 4              리버풀 FC             리버풀 FC   0
# 5            리즈 유나이티드           리즈 유나이티드   0
# 6          맨체스터 시티 FC         맨체스터 시티 FC   0
# 7          맨체스터 유나이티드         맨체스터 유나이티드   0
# 8               번리 FC              번리 FC   0
# 9            브렌트포드 FC           브렌트포드 FC   0
# 10           선덜랜드 AFC           선덜랜드 AFC   0
# 11             아스널 FC             아스널 FC   0
# 12          애스턴 빌라 FC          애스턴 빌라 FC   0
# 13             에버턴 FC             에버턴 FC   0
# 14       울버햄튼 원더러스 FC       울버햄튼 원더러스 FC   0
# 15         웨스트햄 유나이티드         웨스트햄 유나이티드   0
# 16              첼시 FC              첼시 FC   0
# 17        크리스탈 팰리스 FC        크리스탈 팰리스 FC   0
# 18            토트넘 홋스퍼            토트넘 홋스퍼   0
# 19              풀럼 FC              풀럼 FC   0