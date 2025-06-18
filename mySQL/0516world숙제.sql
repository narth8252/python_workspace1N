use world;
SHOW TABLES;
select * from city as c;
SELECT * FROM world.city;
-- 기본 SQL 문제 10문제 (JOIN·서브쿼리 제외)
-- 국가 테이블에서 모든 컬럼을 조회하세요.
SELECT * FROM world.country;
-- 인구(population)가 1억 이상인 나라의 이름과 인구를 조회하세요.
SELECT name, population FROM world.country where population>=100000000;
-- 아시아(Asia) 대륙Continent에 속한 나라들의 이름을 조회하세요.
SELECT name FROM world.country where Continent='asia';

-- 도시(city) 테이블에서 인구가 100만 이상인 도시 이름과 인구를 조회하세요.
SELECT name, population FROM world.city where population>=1000000;
-- 독일(Germany)에 속한 도시(city)들의 이름을 조회하세요.
-- SELECT name FROM world.city where 
-- 국가 이름이 ‘A’로 시작하는 나라들을 조회하세요.
SELECT name FROM world.country where Name like 'A%';
-- 수도(capital) 코드가 100 이상인 국가의 이름과 수도 코드를 조회하세요.
SELECT name, Capital FROM world.country where Capital>=100;
-- 도시(city) 테이블에서 가장 인구가 많은 5개 도시의 이름과 인구를 조회하세요.
SELECT name, Population FROM world.city order by Population desc limit 0,5;
-- 인구(population)가 8,000만 명 이상인 국가의 이름(Name)과 인구(Population)를 조회하세요. 
SELECT name, Population FROM country where population>=80000000;

-- 기본 SQL 문제 10문제 (JOIN·서브쿼리 로만)