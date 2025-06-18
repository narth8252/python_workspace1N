from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
# SQLAlchemy가 PyMySQL을 내부적으로 사용하며 pool 지원
#이엔진을 다른데서 쓸수있는지? 전역으로 쓰고싶으면 앞에 the나 global을 많이 붙임.
theEngine = create_engine(
    "mysql+pymysql://root:1234@localhost:3306/project1",
    pool_size=10,         # 최대 연결 수
    max_overflow=5,       # 초과 시 추가 연결 수
    pool_recycle=3600     # 재활용 시간
)

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# SQLAlchemy가 PyMySQL을 내부적으로 사용하며 pool 지원
theEngine = create_engine(
    "mysql+pymysql://root:1234@localhost:3306/mydb",
    pool_size=10,         # 최대 연결 수
    max_overflow=5,       # 초과 시 추가 연결 수
    pool_recycle=3600     # 재활용 시간
)
