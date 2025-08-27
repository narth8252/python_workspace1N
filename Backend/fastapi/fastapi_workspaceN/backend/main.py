# 250813 AM9:30 풀스택(백엔드-sql-프론트엔드)
# FastAPI를 사용하여 간단한 웹 서버를 구축하는 코드
#실행방법
# cd backend
# conda activate backend
# uvicorn main:app --reload --port 8000
# http://127.0.0.1:8000/docs

# python -m uvicorn main:app --reload --port 8000
# --reload는 개발 중 파일 변경을 감지하고 서버를 자동으로 재시작하는 기능입니다. 
# 이 옵션을 빼면 서버는 한 번 실행된 후 코드가 변경되어도 재시작되지 않습니다. 
# 실제 운영(production) 환경에서는 --reload를 사용하지 않는 것이 일반적입니다.
# http://127.0.0.1:8000/docs#/board/board_index_board__get

# CORS(Cross-Origin Resource Sharing) 설정
# 풀스택개발 - 프론트앤드(사용자인터페이스, 눈에보이는 부분만 담당, html, css,javascript-react등)
#           백앤드-Restful API 서버 - 데이터를 보통 json형태로 응답해주는 서버이다.
#           fastapi등, 장고, 플라스크, 스프링, php, nodejs 등
# 백앤드와 프론트앤드는 별도의 서버이다. 서로간에 자바스크립트를 통해 데이터를 주고 받아야 하는데
# 서로 다른 사이트 (도메인, 아이피, 포트번호만 달라도 서로 다른사이트이다)
# CORS(왜 남의 사이트에서 내사이트를 오는데 라는 오류임)
# 백엔드에서 이 문제를 해결해야함. 특정IP나 Port번호는 내정보를 가져갈수있어라고 허락해야함.
# CORSMiddleware를 통해 이걸 열어준다
# 프론트쪽에 프록시서버를 만들어 접근가능하다는 사람도 있지만 쓰지마(보안취약) -> react는 허용안함. vuejs는 아직허용중인데 고려하지말자. 보

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 애플리케이션 초기화
app = FastAPI()

# React 개발 서버(localhost:3000 등)와 통신하기 위해 필요합니다.
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://www.sessac.com:5173"
]
# 미들웨어:중간에 거쳐간다. 클라이언트===>미들웨어1=>미웨2=>미웨3....===>서버
# CORSMiddleware를 앱에 추가: 
# 나랑 도메인 또는 IP와 port번호 다르면 다른서버(Browser제외)=> 자바스크립트로 남의 사이트 접근하지 말아라
# 가끔 예외있는데 
# 그래서 아래 미들웨어 추가함. 
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

#라우터 연결하기 
from routers import board , score

app.include_router(board.router)  # http://127.0.0.1:8000/board ~~ => board.py가 처리한다  
app.include_router(score.router)

