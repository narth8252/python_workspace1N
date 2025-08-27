# 250814 AM9시 복습: 풀스택(백엔드-sql-프론트엔드)
# FastAPI를 사용하여 간단한 웹 서버를 구축하는 코드
#실행방법
# cd backend2 (작업폴더명)
# conda activate backend (가상환경)
# uvicorn main:app --reload --port 8000
# http://127.0.0.1:8000/docs
# python -m uvicorn main:app --reload

# conda install python-multipart
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

import os, shutil
from typing import Optional #(객체)최근 언어들 경향이 null이나 None값에 대한 처리를 철저하게 하기 위해 만든라이브러리
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
"""
http프로토콜에 정의한 내용
get 방식 - 2048바이트 미만의 텍스트
post방식 - url-encoding방식, FormData방식(파일업로드), raw(json)
        특별히 요청안하면 raw(json), 파일업로드처리 하려면 Form객체통해 데이터 수신해야만한다.
이미지 업로드하고나면 업로드 경로를 아래처럼 접근하고싶다면,
http://127.0.0.1:8000/image_url/파일명 
언어불문, 물리적경로를 url로 바꾸는 방법이 필요
"""
# FastAPI 애플리케이션 초기화
app = FastAPI()

#파일업로드하기(전역변수) - 모든라우터가 공유해야할 변수가 있을때
my_global_settings={
    "api_key":"1203ue", 
    "db_url":"",
    "UPLOAD_DIRECTORY":"./upload_files"
}

#1.업로드 디렉토리가 없을 경우, 디렉토리 만들자
if not os.path.exists(my_global_settings["UPLOAD_DIRECTORY"]):
    os.makedirs(my_global_settings["UPLOAD_DIRECTORY"])
#2./upload_files=>url로 바꾸는 작업필요
#정적디렉토리??
app.mount("/static", 
          StaticFiles(directory=my_global_settings["UPLOAD_DIRECTORY"]), 
          name="static")

# React 개발 서버(localhost:3000 등)와 통신하기 위해 필요합니다.
origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://www.sessac.com:5173"
]
# 미들웨어:중간에 거쳐간다. 클라이언트===>미들웨어1=>미웨2=>미웨3....===>서버
#CORSMiddleware를 앱에 추가: 
# 나랑 도메인 또는 IP와 port번호 다르면 다른서버(Browser제외)=> 자바스크립트로 남의 사이트 접근하지 말아라
# 가끔 예외, 
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
    return {"message":"250814 Hello FastAPI "}

#라우터 연결하기 
from routers import board

#모듈과 모듈간에 전역변수는 원칙적으로 없다. 
#전달 
#Dependency Injection - 의존성 강제주입 
board.settings_container["settings"] = my_global_settings
app.include_router(board.router)  # http://127.0.0.1:8000/board ~~ => board.py가 처리한다  
# app.include_router(score.router)

# http://127.0.0.1:8000/static/my_photo.jpg
#실행방법
# conda activate backend 
# python -m uvicorn main:app --reload  --port 8000

#cmd창을 관리자권한으로 열기 
#conda install pymysql sqlalchemy 
# conda install python-multipart  #파일업로드라이브러리 