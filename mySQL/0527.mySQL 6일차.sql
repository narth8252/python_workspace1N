# user03 사용자새로생성하고 project1디비만들고 권한부여하고 이거로 염alter
-- 회원테이블만들기
-- mysql은 auto_increment 속성있는필드가 무조건 primary key돼야함.
use project1;
create table tb_member(
	member_id bigint auto_increment primary key,
	user_id varchar(40), #이멜을 id하는경우많아서 길게해줘
    password varchar(300), #비번도 200개 넘게나와서 크게해줘
                           #md5암호화 알고리즘써서 암호화해서 저장,
                           #암호화원상복귀(복호화)가 불가능 알고리즘
                           #MD5는 보안에 취약함. 실제 서비스에서는 bcrypt, PBKDF2, Argon2 등을 권장
                           #복호화 불가능한 단방향 해시 알고리즘이므로 VARCHAR(300)은 넉넉히 설정됨
	user_name varchar(40),
	email VARCHAR(100),
    phone varchar(40),
    regdate datetime
);

select * from tb_member; 
#이 SQL 구문은 tb_member 테이블에 있는 **모든 열(all columns)**과 **모든 행(all rows)**을 조회합니다.

insert into tb_member(user_id, password, user_name, email, phone,regdate) values('test1', '1234', '홍길동',
'hong@daum.net', ' 010-0000-0001', now());


#MySQL에서 단방향 암호화
#예:MD5,SHA−1,MySQL의<code>PASSWORD(함수등)사용시, 암호화된값은 복호화X
#MySQL에서 단방향암호화(해시함수)사용시 복호화불가, 비밀번호검증시 입력값암호화하여 비교하는방식으로 인증진행. 
#양방향 암호화 사용시 암호화키관리에 각별히 주의.
#정확한 암호화방식 선택과 컬럼길이설정, 그리고 키관리가 매우중요.
	
-- 게시판테이블 만들기
select * from tb_score;

create table tb_score(
	id bigint primary key auto_increment,
    sname varchar(20),
    kor int,
    eng int,
    mat int,
    regdate datetime
    );
    
select * from tb_score;