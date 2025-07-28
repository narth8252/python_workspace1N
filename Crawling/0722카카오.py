#250722 PM4시 쌤PPT.118p(250717딥러닝종합_백현숙.pptx)
# REST API 키: 14a582e9b43f523799ca7d3d86273c64
# 앱 > 제품 설정 > 카카오맵 > 사용설정 ON 
# https://developers.kakao.com/console/app/1256561/product/kakaoMap

#requests.get으로 가져오는데 
# KakaoAK 14a582e9b43f523799ca7d3d86273c64
#커스텀헤더 키값은 "Authorization: KakaoAK ${REST_API_KEY}" 
# 커스텀헤더 키값은 "Authorization":"KakaoAK c02b3bab10202accf04f99d05b33edf1"



# https://developers.kakao.com/docs/latest/ko/local/dev-guide
# 요청
# curl -v -G GET "https://dapi.kakao.com/v2/local/search/address.json" \
#   -H "Authorization: KakaoAK ${REST_API_KEY}" \
#   --data-urlencode "query=전북 삼성동 100" 

#postman 실행 > New Request >
#get 입력: https://dapi.kakao.com/v2/local/search/address.json?query=전북 삼성동 100
#Headers 입력: 키:Authorization 밸류:KakaoAK 14a582e9b43f523799ca7d3d86273c64
#Params 입력: 키:query 밸류:전북 삼성동 100
# Send누르면 Body에 html코드 출력됨

"""
curl -v -G GET "https://dapi.kakao.com/v2/local/search/keyword. json?y=37.514322572335935&x=127.06283102249932&radius=20000" \
-H "Authorization: KakaoAK ${REST_API_KEY}" \
-- data-urlencode "query=카카오프렌즈"
""" 
url = "https://dapi.kakao.com/v2/local/search/keyword.json?y=37.514322572335935&x=127.06283102249932&radius=20000"
url = url + "&query=카카오프렌즈"
custom_header = {"Authorization":"KakaoAK 14a582e9b43f523799ca7d3d86273c64"}
import requests
import json
response = requests.get(url, headers=custom_header)
if response.status_code == 200:
    data = json. loads(response.text)
    print(data)
else:
    print("error")