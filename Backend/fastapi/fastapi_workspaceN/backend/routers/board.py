from turtle import title
from fastapi import FastAPI 
from fastapi.responses import JSONResponse 

from fastapi import APIRouter, Depends 
from database import Database

router = APIRouter(
    prefix="/board",  #url요청이 /board/~~~~ 로 오는것은 여기서 다 처리한다는 의미임 
    tags=["board"],     #swager문에 표시될 태그임   
    responses= {404:{'decription':'Not found'}} #예외처리 
)

@router.get("/")
def board_index():
    return {"msg":"게시판입니다"}

@router.get("/list")
def board_list():
    with Database() as db_mgr:
        sql = "select * from tb_board" 
        results = db_mgr.executeAll(sql)
    return {"list":results}

from fastapi import Body
import sqlalchemy
@router.post("/insert")
def board_insert(title:str=Body(...), writer:str=Body(...), contents:str=Body(...)):
    sql = """
        insert into tb_board (title, writer, contents, wdate, hit)
        value(:title, :writer, :contents, now(), 0)
        """
    params = [{"title":title, "writer":writer, "contents":contents}]
    try:
        with Database() as db_mgr:
            db_mgr.execute(sql, params)
        return {"msg":"등록성공"}
    except sqlalchemy.exc.SQLAlchemy as e:
        return {"msg":"데이터 등록실패"}

# C:\Users\Admin\Documents\GitHub\python_workspace1N\Frontend > 0813DBSchema.sql
# cmd관리자
# C:\Windows\System32> mysql -u root -p
# Enter password: 1234
# ...
# mysql> use mydb;
# Database changed
# mysql> view table tb_board;
# ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near 'view tacle
# mysql> select * from tb_board;
# Empty set (0.00 sec)

# mysql> insert into tb_board(title, contents, writter, wdate, hit)
#     -> values('제목1', '내용1', 'test1', now(), 0);
# ERROR 1054 (42S22): Unknown column 'writter' in 'field list'
# mysql> insert into tb_board(title, contents, writer, wdate, hit)
#     -> values('제목1', '내용1', 'test1', now(), 0);
# Query OK, 1 row affected (0.03 sec)
# ...
# mysql> insert into tb_board(title, contents, writer, wdate, hit)
#     -> values('제목5', '내용5', 'test5', now(), 0);
# Query OK, 1 row affected (0.01 sec)

# mysql> select * from tb_board;
# +----+------------+------------+----------+-----------+--------------+---------------------+------+
# | id | title      | contents   | filename | image_url | writer       | wdate               | hit  |
# +----+------------+------------+----------+-----------+--------------+---------------------+------+
# |  1 | 제목1      | 내용1      | NULL     | NULL      | test1        | 2025-08-13 12:00:26 |    0 |
# |  2 | 제목2      | 내용2      | NULL     | NULL      | test2        | 2025-08-13 12:00:53 |    0 |
# |  3 | 제목3      | 내용3      | NULL     | NULL      | test3        | 2025-08-13 13:03:09 |    0 |
# |  4 | 제목4      | 내용4      | NULL     | NULL      | test4        | 2025-08-13 13:03:25 |    0 |
# |  5 | 제목5      | 내용5      | NULL     | NULL      | test5        | 2025-08-13 13:03:43 |    0 |
# |  6 | 제목입니다 | 내용입니다 | NULL     | NULL      | 작성자입니다 | 2025-08-13 13:20:31 |    0 |
# +----+------------+------------+----------+-----------+--------------+---------------------+------+
# 6 rows in set (0.00 sec)