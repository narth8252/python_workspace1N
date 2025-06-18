from DBModule import Database

def output():
    db = Database()
    sql = "select * from tb_member"
    rows = db.executeAll(sql)
    for row in rows:
        print(row)
    db.close()

#아이디 중복체크 함수
def idcheck(user_id):
    if user_id == "" or user_id=="test": #예시로 하심:에러체크는 가급적 위에서 해라
        return False #if문 써서 id에 공백오면 False오면 사용불가
    db = Database()
    sql = "select count(*) as cnt from tb_member where user_id = %s" #null이나 no해도되는됨
    row = db.executeOne(sql, (user_id))
    cnt = row["cnt"]
    db.close()
    if cnt ==0:
        return True #중복안됐으니까 쓸수있다.
    return False
    #result = db.executeOne(sql, (user_id,))
    #db.close()
    #return result['cnt'] > 0


#회원가입 절차
def member_register(): 
    db = Database() #객체생성-> DB연결, 이건Database 객체임
    #Database()는 커넥션과 커서포함한 내가만든 DB클래스

    user_id = input("아이디 : ") #input()을 통해 사용자 입력받고
    #if idcheck(user_id): #호출 → 중복 확인
    if not idcheck(user_id):
        print("이미 존재하는 아이디입니다.") #True면 안써도 알아들음
        # db.close()
        return
    print("사용가능한 아이디입니다.")

    password = input("패스워드 : ")
    user_name = input("이름 : ")
    email = input("이메일 : ")
    phone = input("전화 : ")
    sql = """
    insert into tb_member(user_id, password, user_name, email, phone, regdate) 
    values(%s, %s, %s, %s, %s, now()) #NOW()는 현재 시간을 regdate에 자동 기록
    """
    db.execute(sql, (user_id, password, user_name, email, phone))
    db.close()
    print("회원가입이 완료되었습니다.")

#아이디 중복체크 -> 아이디 입력받고나서 디비에 이미 존재하는지,
#존재하면 이미 존재하는 아이디입니다. 하고 함수종료
#사용가능한 아이디입니다. 출력하고 나머지 입력받아 회원가입

if __name__ == "__main__":
    member_register()
    output()