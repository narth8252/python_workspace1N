# 250812 PM3시 25082FastAPI.pptx 
# cd C:\Users\Admin\Documents\GitHub\python_workspace1N\Backend\fastapi\fastapi_workspaceN
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 애플리케이션 초기화
app = FastAPI()

# CORS(Cross-Origin Resource Sharing) 설정
# React 개발 서버(localhost:3000 등)와 통신하기 위해 필요합니다.
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

# CORS 미들웨어를 앱에 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

#@-데코레이터   app
@app.get("/")
def index():
    return {"message":"Hello FastAPI"}

# 쿼리 파라미터로 값 받기1
#http://127.0.0.1:8000/add?x=4&y=8
@app.get("/add")
def add(x:int, y:int):
    return {"x":x, "y":y, "result":x+y}

# 쿼리 파라미터로 값 받기2
#http://127.0.0.1:8000/add2/x/y
@app.get("/add2/{x}/{y}")
def add(x:int, y:int):
    return {"x":x, "y":y, "result":x+y}

#더미데이터
scoreList= [
        {"name":"홍길동", "kor":100, "eng":100, "mat":100, "total":300, "avg":100},
        {"name":"임꺽정", "kor":90, "eng":90, "mat":90, "total":270, "avg":90},
        {"name":"장길산", "kor":80, "eng":80, "mat":80, "total":240, "avg":80}
]

# 성적 목록 반환
@app.get("/scoreList")
def getScoreList():
    # 데이터베이스에서 데이터를 읽어오는 것을 가정
    return {"scoreList":scoreList}

from fastapi import Body 
@app.post("/score/insert" )
def score_insert(name:str = Body(..., embed=True),
                 kor:int = Body(..., embed=True),
                 eng:int = Body(..., embed=True),
                 mat:int = Body(..., embed=True) 
                 ):
    score = {"name":name, 
             "kor":kor, 
             "eng":eng, 
             "mat":mat,
             "total":kor+eng+mat,
             "avg":(kor+eng+mat)/3}
    scoreList.append(score)
    return score 
    # 성적데이터를 처리할 때는 int보다 float으로 지정하는 것이 가장 안전하고 정확합니다.

# 실행방법(코드바꿀때마다 재실행)
# cd fastapi_workspaceN
# conda activate backend (오늘만든 가상환경 실행)
# python -m uvicorn main:app --reload
# http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/docs#/default/add_add_get 
#위사이트에서 보면 백엔드랑 프론트엔드 개발자가 함께 보면서 얘기하면 되게 설명돼있음
#플라스크보다 FastAPI가 편하고 개발자입장에서 좋음


#가상환경설치 및 라이브러리 설치
# conda create –n backend
# conda activate backend
# conda install fastapi
# conda install uvicorn

# 코딩 파트너
# FastAPI와 React 프로젝트를 함께 실행하고 계시는군요. 파일 한 곳만 F5를 누르면 되는 것이 아니라, 백엔드 서버와 프런트엔드 서버를 각각 실행해야 합니다.
# F5 키는 보통 Visual Studio Code에서 현재 열려 있는 파일을 디버깅 모드로 실행할 때 사용됩니다. 하지만 이 경우에는 두 개의 다른 서버를 켜야 해요.
# 1. 백엔드 서버 (FastAPI) 실행하기
# (backend) C:\Users\Admin\Documents\GitHub\python_workspace1N\Backend\fastapi\fastapi_workspaceN>uvicorn main:app --reload
# FastAPI 서버를 실행하는 터미널에서 uvicorn 명령어를 입력해야 합니다.
# 위치: FastAPI 프로젝트 폴더 (예: C:\Users\Admin\Documents\GitHub\python_workspace1N\Backend\fastapi\fastapi_workspaceN)
# 명령어: uvicorn main:app --reload
# 역할: 이 명령어를 실행하면 localhost:8000에서 백엔드 API가 작동하기 시작합니다.

# 2. 프런트엔드 서버 (React) 실행하기
# React 개발 서버를 실행하는 터미널에서 npm 명령어를 입력해야 합니다.
# 위치: React 프로젝트 폴더 (예: C:\Users\Admin\Documents\GitHub\python_workspace1N\Frontend\React\react_project\myhome_vite)
# 명령어: npm run dev
# 역할: 이 명령어를 실행하면 localhost:5173에서 React 웹사이트가 실행됩니다.

# 요약하자면, 터미널 창을 두 개 열어서 각각 백엔드와 프런트엔드 서버를 따로 실행해야 합니다. ---

# 실행 순서
# 터미널 1에서 FastAPI 프로젝트 폴더로 이동한 후, uvicorn main:app --reload를 실행합니다.
# 터미널 2를 새로 열어 React 프로젝트 폴더로 이동한 후, npm run dev를 실행합니다.
# 브라우저에서 localhost:5173에 접속하여 프런트엔드 웹사이트를 확인합니다. '성적처리' 링크를 클릭하면 백엔드 서버에서 가져온 데이터가 화면에 표시될 것입니다.

#백엔드-fastAPI의 main.py를 수정하고, FastAPI Sxagger UI 웹에 가서 POST에 score누르고 Try하고 생겨이씨ㅣ는 곳에 이름,성적입력후
#프론트엔드-Vite+React 웹에 가서 성적처리 클릭하면 추가돼있음.