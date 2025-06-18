from DBModule import Database

def output():
    db = Database()
    sql = "select * from tb_score"
    rows = db.executeAll(sql)
    for row in rows:
        print(row)
    db.close()


#회원가입 절차
def score_register(): 
    db = Database() #객체생성-> DB연결, 이건Database 객체임
    #Database()는 커넥션과 커서포함한 내가만든 DB클래스

    # sname = input("아이디 : ") #input()을 통해 사용자 입력받고
    # #if idcheck(sname): #호출 → 중복 확인

    sname = input("이름 : ")
    kor = input("국어 : ")
    eng = input("영어 : ")
    mat = input("수학 : ")
    sql = """
    insert into tb_score(sname, kor, eng, mat, regdate) 
    values(%s, %s, %s, %s, now()) #NOW()는 현재 시간을 regdate에 자동 기록
    """
    db.execute(sql, (sname, kor, eng, mat))
    db.close()
    print("성적입력이 완료되었습니다.")

#아이디 중복체크 -> 아이디 입력받고나서 디비에 이미 존재하는지,
#존재하면 이미 존재하는 아이디입니다. 하고 함수종료
#사용가능한 아이디입니다. 출력하고 나머지 입력받아 회원가입

if __name__ == "__main__":
    score_register()
    output()