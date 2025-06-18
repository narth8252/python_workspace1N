from sqlalchemy import text
from ScoreData import ScoreData
from DBengine import theEngine  # 누락된 import 추가

class ScoreManager:
    def __init__(self):
        self.scoreList = []

    def output(self):
        sql = "select * from tb_score"
        self.getList(sql)
        #쿼리가 복잡하니까 함수를 쪼개서 작업해야 클린코드
        #(1가지기능에 집중:데이터,쿼리,출력따로만들어서 결합)
        for s in self.scoreList:
            s.output() 

    def getList(self, sql):
        self.scoreList =[]
        with theEngine.begin() as conn:
            result = conn.execute(text(sql))
            for r in result.mappings().all():
                s = ScoreData(r["sname"], r["kor"], r["eng"], r["mat"])
                self.scoreList.append( s )

#데이터 등록하기
    def insertMain(self):
        s = ScoreData()
        s.sname = input("이름 : ")
        s.kor = input("국어 : ")
        s.eng = input("영어 : ")
        s.mat = input("수학 : ")
        sql = """
            insert into tb_score(sname, kor, eng, mat)
            values(:sname, :kor, :eng, :mat)
        """
        params = [{"sname":s.sname, "kor":s.kor, "eng":s.eng, "mat":s.mat}]
        self.insert(sql, params)

    def insert(self, sql, params):
        with theEngine.begin() as conn:
            conn.execute(text(sql), params)

    #평균 통계내기 총인원,수우미양가
    def statistic(self):
        sql="""
            select '전체' grade , count(*) cnt 
            from tb_score 
            union all 
            select grade, count(*) cnt
            from(
                select id, case  when (kor+eng+mat)/3>=90 then '수'
                                    when (kor+eng+mat)/3>=80 then '우' 
                                    when (kor+eng+mat)/3>=70 then '미'
                                    when (kor+eng+mat)/3>=60 then '양'
                                    else '가' 
                        end as grade 
                from tb_score
            ) A
            group by grade
            order by field(grade, '전체', '수', '우', '미', '양', '가')
        """
        temp={"전체":0, "수":0, "우":0, "미":0, "양":0, "가":0}
        with theEngine.connect() as conn:
            result = conn.execute(text(sql))
            for r in result.mappings().all():
                #print(r)
                temp[r["grade"]] = r["cnt"] # 여기를 반드시 실행해야 temp가 채워짐
        
        print("전체",temp["전체"])
        print("수",temp["수"])
        print("우",temp["우"])
        print("미",temp["미"])
        print("양",temp["양"])
        print("가",temp["가"])        
        
        # print("\n[등급별 인원 통계]")
        # for g in ["전체", "수", "우", "미", "양", "가"]:
        #     print(f"{g}: {temp[g]}명")
                
if __name__ == "__main__":
    sm = ScoreManager()
    #sm.insertMain()
    #sm.output()
    sm.statistic()

# #통계평균내기: 전체인원, 수우미양가 각각몇명씩인지
#     def count_grades(self):
#         grade_count = {"수":0, "우":0, "미":0, "양":0, "가":0}
#         for s in self.scoreList:
#             avg = (int(s.kor)+int(s.eng)+int(s.mat)) / 3
#             grade = self.get_grade(avg) #이미만든 함수사용
#             grade_count[grade] += 1
#         return grade_count
#     def output(self):
#         sql = "select * from tb_score"
#         self.getList(sql)

#         for s in self.scoreList:
#         avg = (int(s.kor) + int(s.eng) + int(s.mat)) / 3
#         grade = self.get_grade(avg)
#         print(f"{s.sname} | 국어: {s.kor}, 영어: {s.eng}, 수학: {s.mat}, 평균: {avg:.2f}, 등급: {grade}")

#         self.stats()

#         grade_count = self.count_grades()
#         print("\n[등급별 인원 통계]")
#         for grade in ['수', '우', '미', '양', '가']:
#             print(f"{grade}: {grade_count[grade]}명")

 



