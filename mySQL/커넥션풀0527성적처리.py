# conda activate myenv1
# pip install sqlalchemy 

# TERMINAL 에 커맨드창에 위에꺼 복사해서 쳐넣으면 아래처럼나옴
# (base) C:\Users\Administrator\Documents\python_workspace1N\mySQL>conda activate myenv1
# (myenv1) (base) C:\Users\Administrator\Documents\python_workspace1N\mySQL>pip install sqlalchemy
# C:\Users\Administrator\AppData\Roaming\MySQL\Workbench 여기 설치됨?
 
#챗GPT말고 구글링 "sqlalchemy tutorial" 1페이지에서 찾아라.
#https://soogoonsoogoonpythonists.github.io/sqlalchemy-for-pythonist/tutorial/1.%20%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC%20%EA%B0%9C%EC%9A%94.html
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

#SQLAlchemy가 PyMySQL을 내부적으로 사용하며 Connection Pool 지원
engine = create_engine(
    "mysql+pymysql://root:1234@localhost/project1",
    pool_size=10,         # 최대 연결 수
    max_overflow=5,       # 초과 시 추가 연결 수
    pool_recycle=3600     # 재활용 시간
)

try:
    conn = engine.connect()
    result = conn.execute("SELECT DATABASE();")
    print("연결 성공, 현재 DB:", result.scalar())
except Exception as e:
    print("연결 실패:", e)
finally:
    conn.close()

try:
    conn = engine.connect()
    print("데이터베이스 연결 성공")
except SQLAlchemyError as e:
    print("데이터베이스 연결 실패:", e)


#2.0ver 이전엔 conn.execute("쿼리")
result = conn.execute(text("SELECT * FROM tb_score"))

# for row in result:
#     print(row)

#dict타입으로 출력
rows = result.mappings().all()
for row in rows:
        print(dict(row))

conn.close()

#데이터 추가하기
conn = engine.connect()
sql = text("""
        insert into tb_score (tb_score, sname, kor)
        values(:tb_score, :sname, :kor)
""")
conn.execute(sql,[{"tb_score": "sname":"커넥션풀1", "kor":70, "eng":60},{"tb_score":"sname":"커넥션풀2", "eng":50,"kor":40}])
conn.commit()
conn.close()

# "C:\Users\Administrator\Documents\python_workspace1N\mySQL\커넥션풀1.py"