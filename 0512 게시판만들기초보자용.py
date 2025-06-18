#0512 게시판만들기
#회원관리 - 회원번호,아이디,password, 이름,전화번호,이메일
#게시판 - 회원번호, 글번호, 제목,내용, 작성일,조회수
#회원가입, 수정,탈퇴, 조회(자기정보보여주기:아이디,패스워드 입력하면)
#DB없는상태로 join함수없이
#게시글작헝 - 회원번호, 제목,내용, 작성일(date)gpt물어봐, 조회수0
  #읽어보기
  #수정(글쓴이만) 회원번호랑 패스워드 입력하면
  #삭제(글쓴이만) 회원번호랑 패스워드 입력하면

class Member:
    def __init__(self):
        self.users = {}  # {"아이디": "비밀번호"}

    def signup(self): #회원가입
        user_id = input("아이디 입력: ")
        if user_id in self.users:
            print("이미 존재하는 아이디입니다.\n")
            return
        password = input("비밀번호 입력: ")
        self.users[user_id] = password
        print("회원 가입 완료!\n")

    def login(self): #로그인
        user_id = input("아이디 입력: ")
        password = input("비밀번호 입력: ")
        if self.users.get(user_id) == password:
            print(f"로그인 성공! {user_id}님 환영합니다.\n")
            return user_id
        else:
            print("로그인 실패. 아이디 또는 비밀번호가 틀립니다.\n")
            return None


class Board:
    def __init__(self):
        self.posts = []  # 게시글 저장
        self.logged_in_user = None

    def write_post(self, user_id):
        title = input("제목을 입력하세요: ")
        content = input("내용을 입력하세요: ")
        post = {"번호": len(self.posts) + 1, "작성자": user_id, "제목": title, "내용": content}
        self.posts.append(post)
        print("게시글 등록 완료!\n")

    def list_posts(self):
        if not self.posts:
            print("게시글이 없습니다.\n")
            return
        print("\n게시글 목록:")
        for post in self.posts:
            print(f"{post['번호']}. [{post['작성자']}] {post['제목']}")
        print()


def main():
    member = Member()
    board = Board()

    while True:
        print("=== 메인 메뉴 ===")
        print("1. 회원가입")
        print("2. 로그인")
        print("3. 종료")
        choice = input("선택 (1~3): ")

        if choice == "1":
            member.signup()
        elif choice == "2":
            user = member.login()
            if user:
                while True:
                    print(f"--- 게시판 ({user}) ---")
                    print("1. 글 쓰기")
                    print("2. 글 목록 보기")
                    print("3. 로그아웃")
                    post_choice = input("선택 (1~3): ")

                    if post_choice == "1":
                        board.write_post(user)
                    elif post_choice == "2":
                        board.list_posts()
                    elif post_choice == "3":
                        print("로그아웃 되었습니다.\n")
                        break
                    else:
                        print("잘못된 선택입니다.\n")
        elif choice == "3":
            print("프로그램 종료합니다.")
            break
        else:
            print("잘못된 선택입니다.\n")


# 프로그램 실행
main()


#write_post()	제목과 내용을 입력받아 게시글 리스트에 추가
#list_posts()	저장된 게시글 제목을 번호와 함께 출력
#run()	메뉴 반복해서 보여주고, 선택에 따라 기능 실행

#   def autosumself():
#     text = input("self.)")
#     textList  = text.split(",")
#     textList = [i[:(i.index("=")-1)] for i in textList]
#     textList = [i.strip() for i in textList] 
#     output = []
#     for i in textList:
#         output.append("self."+ i + " = " + i)
        
#     print("========절취선==========")
#     for o in output:
#         print(o)
#     print("========절취선==========")
# autosumself()
