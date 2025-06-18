from DBModule import Database

class ScoreData:
    def __init__(self, sname="", kor=0, eng=0, mat=0, total=0, average=0):
        self.sname =
        self.kor = kor
        self.eng = eng
        self.mat = mat
        self.total = total
        self.average = average

    def output(self):
        print(self.sname, self.kor, self.eng, self.mat, self.total, self.average)

class ScoreManager:
    def __init__(self):
        self.dataList = []

    def append(self):
        s = ScoreData()
        s.sname = input("이름 : ")
        s.kor = int(input("국어 : "))
        s.eng = int(input("영어 : "))
        s.mat = int(input("수학 : "))

        sql = """
        insert into tb_score(sname, kor, eng, mat, regdate) 
        values(%s, %s, %s, %s, now())
        """
        db = Database()
        db.execute(sql, (s.sname, s.kor, s.eng, s.mat))
        db.close()

    def output(self):
        sql = """
        select sname, kor, eng, mat,
        (kor+eng+mat) as total, 
        (kor+eng+mat)/3 as average
        from tb_score
        """
        db = Database()
        rows = db.executeAll(sql)
        self.dataList = []

        for r in rows:
            s = ScoreData(r['sname'], r['kor'], r['eng'], r['mat'], r['total'], r['average'])
            self.dataList.append(s)

        for s in self.dataList:
            s.output()

if __name__ == "__main__":
    sm = ScoreManager()
    sm.append()

"""
-커넥션풀 -> DB연결-데이터읽고쓰기-DB연결끊기
		이렇게 하면 연결과 끊기가 시간이 더 많이 걸린다.
        DB연결자를 50개쯤 만들어놓고 : connection개체를 많이 만들어놓고 끊기없이 돌려쓰는 법
		연결자50개를 DB접속할때마다 연결자를 50개씩 시스템다운됨
        라이브러리는 싱글톤으로 만들어져있음.
        직접설계해서 사용하다가 요즘엔 지원하는 라이브러리들이 있음.(MS)
        시중에 커넥션풀 많이 없음
-기존 방식: 요청마다 DB 연결 → 쿼리 실행 → 연결 종료
: 많은 트래픽이 오면 DB에 연결 요청 폭주 → 지연/오류 발생

-커넥션 풀 Connection Pool: DB연결을 미리 여러개 열어두고(풀링)
→ 필요한 스레드가 가져다 쓰고, 사용 후 반납
→ 재사용 가능, 연결 비용 줄어듦

이 클래스는 반드시 싱글톤방식으로 만들어야한다.
(객체를 반드시 1개만 만드는 클래스설계기법)
생성자에서 못만들게 막고 @classmethod 라는 데코레이터를 이용해서 객체생성했음.
20년 전에는 만들어쓰다가 현재는 별도의 라이브러리를 갖다쓴다.

-pymysql + DBUtils.PooledDB 조합 추천
pip install pymysql dbutils

예시
from DBUtils.PooledDB import PooledDB
import pymysql

class Database:
    pool = PooledDB(
        creator=pymysql,
        maxconnections=10,  # 커넥션 최대 개수
        mincached=2,        # 생성 시 미리 커넥션 2개 확보
        maxcached=5,        # 캐시 가능한 최대 커넥션
        blocking=True,      # 커넥션 없을 때 대기 여부
        host='localhost',
        user='root',
        password='your_password',
        database='your_db',
        charset='utf8mb4'
    )

    def __init__(self):
        self.conn = self.pool.connection()
        self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)

    def execute(self, sql, params=None):
        self.cursor.execute(sql, params)
        self.conn.commit()

    def executeAll(self, sql, params=None):
        self.cursor.execute(sql, params)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()


"""