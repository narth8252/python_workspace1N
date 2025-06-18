# 250528 pm5:30 설치: 파이썬 프롬프트창에
# conda activate myenv1
# pip install DBUtils 
#설치경로 : C:\Users\littl\AppData\Roaming\Python\Python38\site-packages\dbutils
from dbutils.pooled_db import PooledDB
import pymysql

# PooledDB를 이용한 커넥션 풀 구성
pool = PooledDB(
    creator=pymysql,
    maxconnections=10,
    mincached=2,
    blocking=True,
    host='localhost',
    user='root',
    password='1234',
    database='mydb',
    charset='utf8mb4'
)

# 커넥션 얻기(옛스러운데 아직 실무에서 씀)
# 구글링 "python tutorial dbutils tutorial" 
# https://webwareforpython.github.io/DBUtils/main.html 
# 실무에서는 이거랑 챗GPT한테 이거써서 insert 심플하게 만들어달라고 하고 복붙해라.
conn = pool.connection()
cursor = conn.cursor()
cursor.execute("SELECT * FROM tb_score")
print(cursor.fetchall())
cursor.close()
conn.close()