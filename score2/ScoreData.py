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

from sqlalchemy import text
from DBengine import theEngine

#orm사용시 class 만들어놓고 쓰는것이 맞음
class ScoreData:
    #DB에서 레코드셋 가져왔을때, ScoreData객체로 만들어서 따로 관리가능
    #select ... (kor+eng+mat) total
    #s = ScoreData()
    #s = ScoreData("홍길동") 등 8가지 방식의 객체생성 가능해짐.
    def __init__(self, sname="", kor=0, eng=0, mat=0):
                # total=0, average=0, grade=""):
        self.sname = sname
        self.kor = kor
        self.eng = eng
        self.mat = mat
        self.process()
        # self.total = total
        # self.average = average
        # self.grade = grade

    def output(self): #쿼리
        print(f"{self.sname}", end="\t")
        print(f"{self.kor}", end="\t")
        print(f"{self.eng}", end="\t")
        print(f"{self.mat}", end="\t")
        print(f"{self.total}", end="\t")
        print(f"{self.average}", end="\t")
        print(f"{self.grade}")

    def process(self): #쿼리가 힘들면
        self.total = self.kor+self.eng+self.mat
        self.average = self.total/3
        if self.average >= 90:
            self.grade = "수"
        elif self.average >= 80:
            self.grade = "우"
        elif self.average >= 70:
            self.grade = "미"
        elif self.average >= 60:
            self.grade = "양"
        else :
            self.grade = "가"
        

if __name__=="__main__":
    with theEngine.begin() as conn:
        sql ="select * from tb_score"
        result = conn.execute(text(sql))
        for r in result.mappings().all(): #tuple로 가져온다 
            s = ScoreData(r["sname"], r["kor"], r["eng"], r["mat"])
            s.output()
        # print( dict(row))
