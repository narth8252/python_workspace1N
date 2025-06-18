"""
이 두 줄에 빨간 줄이 뜨는 이유
1. SQLAlchemy가 설치되어 있지 않은 경우
아래 명령어를 터미널에서 실행해서 설치해 줘:
pip install SQLAlchemy
또는 conda 쓰고 있으면:
conda install sqlalchemy
2. 인터프리터(환경)가 잘못 지정된 경우
VS Code 하단 왼쪽에 보면 [Python 3.x.x ('env_name')] 같은 인터프리터 표시가 있어.
이게 현재 사용하는 파이썬 가상환경인데, 여기에 SQLAlchemy가 설치된 게 아닐 수 있어.

해결: Ctrl + Shift + P 누르고 → Python: Select Interpreter 입력해서 → SQLAlchemy 설치된 환경 선택
"""
#SQLAlchemy가 PyMySQL을 내부적으로 사용하며 Connection Pool 지원

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

engine = create_engine(
    "mysql+pymysql://root:1234@localhost:3306/mydb",
    pool_size=10,         # 최대 연결 수
    max_overflow=5,       # 초과 시 추가 연결 수
    pool_recycle=3600     # 재활용 시간
)
with engine.connect() as conn:
    sql = """select empno, ename, sal from emp"""
    result = conn.execute(text(sql))
    for row in result.all():
        print(row)


#2.검색어를 전달할때
ename = "조승연" #키보드입력으로 변경
with engine.connect() as conn:
    sql = """
        select empno, ename, sal from emp 
        where ename=:ename
        """
    #name 방법1.
    # result = conn.execute(text(sql), {"ename": ename})
    # rows = result.all()
    # if len(rows) == 0:
    #     print("없음")
    # else:
    #     for row in rows:
    #         print(dict(row._mapping))  # Row 객체를 dict로 변환해 보기 쉽게 출력
    
    #name 방법2.
    result = conn.execute(text(sql), {"ename": ename})
    temp = result.all()

    if len(temp) == 0:
        print("없음")
    else:
        for row in temp:
            print(row) # 또는 print(row['empno'], row['ename'], row['sal'])

#3.insert :SQLAlchemy로 emp테이블에 새사원정보 추가
with engine.connect() as conn:
    sql = """select ifnull(max(empno), 0)+1 
        from emp
        """
    result = conn.execute(text(sql))
    empno = result.all()[0][0]
    sql = """
        insert into emp(empno, ename, sal)
        values(:empno, :ename, :sal)
        """
    conn.execute(text(sql), 
                 [{"empno":empno, 
                   "ename":"홍길동"+str(empno), 
                   "sal":9000}])
    conn.commit() #반드시 커밋

# #3.insert :SQLAlchemy로 emp테이블에 새사원정보 추가
# with engine.connect() as conn:
#     sql = """select ifnull(max(empno), 0)+1 
#         from emp
#         """
#     result = conn.execute(text(sql))
#     empno = result.all()[0][0]
#     sql = """
#         insert into emp(empno, ename, sal)
#         values(:empno, :ename, :sal)
#         """
#     conn.execute(text(sql), 
#                  [{"empno":empno, 
#                    "ename":"홍길동"+str(empno), 
#                    "sal":9000}])
#     conn.commit() #반드시 커밋

# #4-1.insert :SQLAlchemy로 test1테이블에 새정보 추가
# with engine.connect() as conn:
#     sql = """select ifnull(max(id), 0)+1 from test1
#         """
#     result = conn.execute(text(sql))
#     id = result.all()[0][0]
#     sql = """
#         insert into test1 values(:id, :field)
#         """
#     conn.execute(text(sql), #이것도 이유없음.그냥문법임.그대로 써.
#                  [{"id":id, "field":"test"}])
#     conn.commit() #반드시 커밋

# #4-2.insert :SQLAlchemy로 test2테이블에 varchar넘어가는 새정보 추가-트랜젝션처리가 안되는경우(문제되는코드)
# with engine.connect() as conn:
#     sql = """select ifnull(max(id), 0)+1 from test2
#         """
#     result = conn.execute(text(sql))
#     id = result.all()[0][0]
#     sql = """
#         insert into test2 values(:id, :field)
#         """
#     conn.execute(text(sql), #이것도 이유없음.그냥문법임.그대로 써.
#                  [{"id":id, "field":"test12345617"}])
#     conn.commit() #반드시 커밋

#4-3.insert :SQLAlchemy로 test1테이블에 새정보 추가-
# 트랜젝션처리(all or nothing 예매동시되면 늦게한사람의 모든정보취소)
# ACID(시험多)
# Atomicity (원자성)   모든 작업이 전부 수행되거나 전혀 수행되지 않아야 함
# Consistency (일관성)   트랜잭션 전후의 DB 상태는 항상 일관되어야 함
# Isolation (격리성)   동시에 실행되는 트랜잭션은 서로 간섭하지 않아야 함
# Durability (지속성)   성공한 트랜잭션의 결과는 영구적으로 반영되어야 함
with engine.begin() as conn:
    sql = """select ifnull(max(id), 0)+1 from test1
        """
    result = conn.execute(text(sql))
    id = result.all()[0][0]
    sql = """
        insert into test1 values(:id, :field)
        """
    conn.execute(text(sql), #이것도 이유없음.그냥문법임.그대로 써.
                 [{"id":id, "field":"test"}])
    
    sql = """
        insert into test2 values(:id, :field)
        """
    conn.execute(text(sql),
                 [{"id":id, "field":"test"}])
    # conn.commit() #커밋안해도됨.트랜젝션자동처리됨

