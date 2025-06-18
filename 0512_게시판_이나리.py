#회원관리 - 회원번호,아이디,password, 이름,전화번호,이메일
#게시판 - 회원번호, 글번호, 제목,내용, 작성일,조회수
# User: 사용자 정보를 관리하는 클래스
# Board: 게시판의 기본적인 정보를 관리하는 클래스
# Post: 게시글의 정보를 관리하는 클래스
# Comment: 댓글의 정보를 관리하는 클래스


# User: 사용자 정보를 관리하는 클래스
class User:
    def __init__(self):
        self.users = {}     #기능추가1.회원가입(회원정보저장)
        self.user_id = None #현재로그인한 유저
        # self.user_id = [] # 로그인 사용자 저장

    def UserLogin(self):
        user_id = input("아이디 : ")
        password = input("비밀번호 :")
        self.users[user_id] = password
        # 실제 로그인 검증은 생략하고 입력만 받음
        self.user_id = user_id
        print("로그인 완료. 게시글을 작성하세요.")
        return user_id
      
    def register(self):
        new_id = input("새 아이디: ")
        if new_id in self.users:
            print("이미 존재하는 아이디입니다.")
            return
        new_pw = input("비밀번호: ")
        self.users[new_id] = new_pw
        print("회원가입 완료!\n")


# Board: 게시판 관리 클래스
class Board:
    def __init__(self):
        self.post = []   #게시글들이 들어있는 리스트 생성
    
    def write_post(self, user_id):
        title = input("제목입력 : ")
        content = input("내용입력 : ")
        #내용은딕으로   len현재까지저장된게시글수
        post = {"번호":len(self.post) + 1, #게시글을0번이아닌 1부터 시작
                "작성자": user_id, 
                "제목": title,
                "내용": content}
        self.post.append(post) #게시글 더하기 추가
        #__init__() 함수에서 self.posts = []가 정상적으로 선언되지 않음
        print("게시글 등록")

    #게시글리스트 없음 추가 
    def list_post(self):
        print("\n 게시글 목록:  ")
        if not self.post:
            print("게시글 없음\n")
            return
        
        for post in self.post:
            print(f"{post["번호"]}. [{post["작성자"]}] {post["제목"]}")
        print()

#Step 2: 메인 메뉴에 “회원가입” 항목 추가
#실행흐름: 1.회원가입(아이디/비번등록) 2.로그인성공하면 3.글쓰기/글목록 메뉴접근
#글 목록 출력”, “글 삭제”, “로그아웃” 등을 단계적으로 추가할 수 있어요.
def main():
    user = User()
    board = Board()

    while True:
        print("----메인 메뉴----")
        print("1. 회원가입")
        print("2. 로그인")
        print("3. 글쓰기")
        print("4. 나가기")
        choice = input("선택(번호로) : ")

        if choice == "1":
            user.register()

        elif choice == "2":
            user.UserLogin()

        elif choice == "3":
            if not user.user_id:
                print("먼저 로그인하세요.")
                continue

            while True:
                print(f"--- 게시판 ({user.user_id}) ---")
                print("1. 글 쓰기")
                print("2. 글 목록")
                print("3. 이전 메뉴로")
                post_choice = input("선택 (번호로): ")

                if post_choice == "1":
                    board.write_post(user.user_id)
                elif post_choice == "2":
                    board.list_post()
                elif post_choice == "3":
                    break
                else:
                    print("잘못된 선택입니다.\n")

        elif choice == "4":
            print("프로그램 종료합니다.")
            break
        else:
            print("메뉴를 다시 선택해주세요.\n")
#프로그램 실행
main() #위치	main() 정의는 밖에, 호출은 맨 아래에서 수행






 