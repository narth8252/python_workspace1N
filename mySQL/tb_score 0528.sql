use project1;
SELECT * FROM project1.tb_score;

-- 안수현님 
select avg.grade, count(*) as cnt
from (
   select case 
         when (ts.kor + ts.eng + ts.mat)/3 >= 90 then "수"
         when (ts.kor + ts.eng + ts.mat)/3 >= 80 then "우"
         when (ts.kor + ts.eng + ts.mat)/3 >= 70 then "미"
         when (ts.kor + ts.eng + ts.mat)/3 >= 60 then "양"
         else "가"
      end as grade
   from tb_score ts 
   group by ts.id
) as avg
group by avg.grade;

-- 쌤풀이:
SELECT grade, COUNT(*) AS cnt
FROM (
    SELECT '전체' AS grade
    FROM tb_score
    UNION ALL
    SELECT
        CASE  
            WHEN (kor + eng + mat) / 3 >= 90 THEN '수'
            WHEN (kor + eng + mat) / 3 >= 80 THEN '우' 
            WHEN (kor + eng + mat) / 3 >= 70 THEN '미'
            WHEN (kor + eng + mat) / 3 >= 60 THEN '양'
            ELSE '가' 
        END AS grade
    FROM tb_score
) AS derived
GROUP BY grade
ORDER BY FIELD(grade, '전체', '수', '우', '미', '양', '가');

-- union된걸 order