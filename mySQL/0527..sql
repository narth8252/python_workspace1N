#1.데이터베이스: mysql8ver이후부터 한글안깨짐
#root 계정만 DB만들고 계정만들수있음(홈에서 root)
create database project1;

-- 계정만들기
-- user03 계정만들고 비번은 1234이다
create user 'user03'@'localhost' identified by '1234';
-- 계정에게 권한주기 : project1 디비에 접근할
grant all privileges on project1.* to 'user03'@'localhost';
