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
    "mysql+pymysql://root:1234@localhost/mydb",
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
result = conn.execute(text("SELECT * FROM emp"))

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
        insert into emp (empno, ename, sal)
        values(:empno, :ename, :sal)
""")
conn.execute( sql, [{"empno":10001,
                     "ename":"우즈2",
                     "sal":8000},
                     {"empno":10002,
                     "ename":"우즈3",
                     "sal":8000}])
conn.commit()
conn.close()

# "C:\Users\Administrator\Documents\python_workspace1N\mySQL\커넥션풀1.py"

#0528 09:23 파이썬개발자를 위한 SQLAlchemy(연금술):라이브러리 이름임.
# ORM - Object     Relation    Mapping
#     객체지향언어 - 관계형DB - 연결하는프로그램기법
#     프로그램:django(장고, 파이썬 웹프레임워크), JPA-JAVA, 파이썬에서 SQLAlchemy가 지원
#     Table이 10개미만일때는 편함(빌려다돌리는거라 데이터 多으면 힘들어짐)
#     Table 3개이상 join시 속도문제, 쿼리(테이블간의 구조)가 복잡해질수록 ORM만으로 처리힘듬 

# SPA - Single   Page  Application
#     사용사이트: 인스타그램,페이스북 등
#     리액트임. 부드럽게 넘어감. 
#     인스타그램 개발하다가 만들어낸게 "React 라이브러리"인데 그래서 리액트가 핫해짐.(디자인이뻐서)
#     SPA방식웹개발시, 주로 ORM방식 많이사용.
#     웹페이지를 하나만 만들어서 페이지체인지하는 방식으로 "화면이동이 매끄럽고" 무한스크롤 등 지원
#     프로그램: react(국내사용압도적), vuejs(국내금융권,리액트대비쉬움,3.0ver이 리액트같아서 리액트에 먹힐?),
#              angular, polymer(유튜브)

#파이썬개발자를 위한 SQLAlchemy(연금술)>데이터베이스와 연결하기
# sqllite - 본의미는 DB는 아니고 파일이다.
# (DB는 primarykey지정해주면 같은내용넣으면 중복체크해주지만, 파일은 추가하는대로 돼서 개발자가 일일히 체크)
# 
# 트랜잭션 - 2개이상 연산이 성공해야한다. all or nothing(원자성:시험多)
#예약시스템에 꼭 필요! 은행거래, 물건구매, 티켓/숙소예약, 결제시 포인트줬다가 취소하면 포인트회수하는 것
#트랜잭션 묶어놓으면 촤라락 취소해버림.(insert, )
# 예약/구매시 여러명 동시에 부킹하는데 결제빠른사람이 성공하면 중간단계에 있는 사람 DB에 저장된 모든것 취소

# ACID(시험多)
# Atomicity (원자성)   모든 작업이 전부 수행되거나 전혀 수행되지 않아야 함
# Consistency (일관성)   트랜잭션 전후의 DB 상태는 항상 일관되어야 함
# Isolation (격리성)   동시에 실행되는 트랜잭션은 서로 간섭하지 않아야 함
# Durability (지속성)   성공한 트랜잭션의 결과는 영구적으로 반영되어야 함
