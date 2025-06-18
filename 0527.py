
import pymysql

class Database():
    def __init__(self):
        self.db = pymysql.connect(
            host = 'localhost', #ip줘야한다. localhost-loop back 주소, 1237.0.0.1이 같음
            user = 'user03', #계정아이디
            password = '1234', #패스워드
            db = 'project1',#접근할db명
            port = 3306) #port, 프로세스식별값,프로세스안데 소켓이라는 객체有
                        #소켓은 통신담당 라이브러리로 소켓에 부여된번호가 port
                        #2byte 정수1~65535까지 가능, 1~1000은 함부로 못씀
                        #80 - http www.daum.net :80
                        #21-telnet, 22-ssh, 23-ftp

        self.cursor = self.db.cursor(pymysql.cursor.DictCursor)

#insert, update, delete
    def execute(self, query, args=()): #args에 매개변수기본값을 tuple로 받아오게 준것임. *치니까 안되서. 
        #execute써서 db에 넣을거야,
        print(args)
        self.cursor.execute(query, args)
        self.db.commit()
        #한쪽에 모아서 만드는것도 클린코드임

    #데이터 딱 한개만 가져오기 - scalar쿼리포함, select count(*) from tb_member
    def executeOne(self, query, args=()):
        self.cursor.execute(query, args)
        row = self.cursor.fetchone()
        return row #결과반환해야함,첫번째레코드값 하나만 가져간다
    
    #데이터 여러개가져오기
    def executeAll(self, query, args=()):
        self.cursor.execute(query, args)
        rows = self.cursor.fetchall()
        return rows #결과반환해야함, 레코드값 모두다 가져간다.
    
    def close(self):
        if self.db.open:
            self.db.close()
